//! Hand-rolled HTTP/1.1 server plumbing on std::net (ledger decision 1A:
//! no axum/hyper/tokio). Wire behavior is byte-modeled on the goish
//! runtime's net/http port (goish-1.0.0-alpha.14), which the frozen
//! `./laya` binary serves with; every observable rule below was probed
//! against that binary and matched (see wayfinder/research/std-port-serving.md):
//!
//! - Response layout: status line, handler-set headers sorted bytewise by
//!   canonical key, then derived extras in fixed order [Date,
//!   Content-Length, Content-Type, Connection, Transfer-Encoding], CRLF,
//!   body. Date is always emitted (IMF-fixdate) unless the handler set
//!   one. Content-Length iff a body was written (HEAD advertises the
//!   would-be length, body suppressed).
//! - Connection header: "keep-alive" only on HTTP/1.0 + self-delimiting
//!   response; "close" only on HTTP/1.1 when the connection will close
//!   (derived, in the extras block; a handler-set Connection sorts with
//!   the handler block instead, e.g. 417).
//! - Parser limits: 16 KiB per line, 16 KiB+4 KiB total head (431 +
//!   close), 16 MiB buffered body cap. Malformed request line/headers,
//!   bad Content-Length and chunk errors get "400 Bad Request" + close;
//!   unknown transfer coding 501; protocol version other than 1.0/1.1
//!   505; unknown Expect 417. A header line over the per-line cap while
//!   the head stays under the total cap closes the connection silently
//!   at the read deadline (observed goish behavior).
//! - Expect: 100-continue writes "HTTP/1.1 100 Continue\r\n\r\n" before
//!   the body read (proto >= 1.1, body indicated).
//! - Deadlines: 5 s (ReadHeaderTimeout) from a connection's first request
//!   through its buffered head+body read; 30 s (IdleTimeout) between
//!   keep-alive requests. Expiry closes silently — no 408, no bytes.
//! - Body is fully buffered before dispatch (goish alpha.14 semantics);
//!   chunked wins over Content-Length, which is then ignored.
//! - MaxConcurrentConns: at most 32 connections are served; the accept
//!   loop holds a slot across each connection's life, so extras queue in
//!   the kernel listen backlog exactly like the goish server.
use alloc::{
    string::{String, ToString},
    vec::Vec,
};
#[cfg(unix)]
use core::sync::atomic::AtomicI32;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_LINE: usize = 16 * 1024; // MaxHeaderBytes: one line
const MAX_HEAD: usize = 16 * 1024 + 4096; // initialReadLimitSize: total head
const MAX_BODY: usize = 16 * 1024 * 1024; // parser cap; the app enforces 1 MiB
const READ_HEADER_TIMEOUT: Duration = Duration::from_secs(5);
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_CONNS: usize = 32;
const POLL: Duration = Duration::from_millis(250);
/// Accept-poll granularity: goish blocks in accept (0 ms latency); a
/// non-blocking poll adds at most this much to new connections, kept
/// small enough (5 ms) to stay inside every test timing window.
const ACCEPT_POLL: Duration = Duration::from_millis(5);
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Shared server state: drain flag and in-flight request count. The
/// signal path sets `draining`; the accept loop and connection threads
/// poll it. `active` counts requests between first byte and completed
/// response — keep-alive waits in between are idle and close on drain.
pub struct Shared {
    pub draining: AtomicBool,
    active: AtomicUsize,
}

impl Default for Shared {
    fn default() -> Self {
        Self::new()
    }
}

impl Shared {
    pub fn new() -> Self {
        Self {
            draining: AtomicBool::new(false),
            active: AtomicUsize::new(0),
        }
    }
    fn enter(&self) {
        self.active.fetch_add(1, Ordering::AcqRel);
    }
    fn leave(&self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
    fn active(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }
}

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    /// n for fixed-length, -1 for chunked, 0 for none.
    pub content_length: i64,
    pub body: Vec<u8>,
    pub http10: bool,
    /// False when the request asks to close (1.1 "close", or 1.0 without
    /// "keep-alive").
    pub keep_alive: bool,
}

impl Request {
    pub fn header(&self, name: &str) -> &str {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
            .unwrap_or("")
    }
}

/// Go MIME canonicalization: upper-case the first letter and any letter
/// after '-', lower-case the rest ("WWW-Authenticate" -> "Www-Authenticate").
pub fn canonical_key(k: &str) -> String {
    let mut upper = true;
    k.chars()
        .map(|c| {
            let out = if upper {
                c.to_ascii_uppercase()
            } else {
                c.to_ascii_lowercase()
            };
            upper = c == '-';
            out
        })
        .collect()
}

const EXTRA_ORDER: [&str; 5] = [
    "Date",
    "Content-Length",
    "Content-Type",
    "Connection",
    "Transfer-Encoding",
];

/// The response accumulator the handler drives; mirrors goish's
/// ResponseWriter (Header().Set / WriteHeader / Write) for the app's
/// single-shot reply pattern.
pub struct Response {
    headers: Vec<(String, String)>,
    status: Option<u16>,
    pub body: Vec<u8>,
    keep_alive: bool,
    http10: bool,
    is_head: bool,
}

impl Response {
    pub fn new(request_method: &str, http10: bool, keep_alive: bool) -> Self {
        Self {
            headers: Vec::new(),
            status: None,
            body: Vec::new(),
            keep_alive,
            http10,
            is_head: request_method == "HEAD",
        }
    }
    pub fn set_header(&mut self, key: &str, value: &str) {
        let key = canonical_key(key);
        if let Some(slot) = self.headers.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value.to_string();
        } else {
            self.headers.push((key, value.to_string()));
        }
    }
    fn header_is_set(&self, name: &str) -> bool {
        self.headers.iter().any(|(k, _)| k == name)
    }
    pub fn write_header(&mut self, code: u16) {
        if self.status.is_none() {
            self.status = Some(code);
        }
    }
    pub fn write(&mut self, data: &[u8]) {
        self.body.extend_from_slice(data);
    }
    /// Render status line + headers + body exactly as goish writes them.
    /// Returns (bytes, keep_alive).
    fn render(self) -> (Vec<u8>, bool) {
        let code = self.status.unwrap_or(200);
        let status_line = crate::format!(
            "HTTP/{} {} {}\r\n",
            if self.http10 { "1.0" } else { "1.1" },
            code,
            status_text(code)
        );
        let mut keep_alive = self.keep_alive;
        let connection_handler_set = self.header_is_set("Connection");
        let mut connection: Option<&'static str> = None;
        if self.body.is_empty() && !self.is_head && !self.header_is_set("Content-Length") {
            // No body written and none delimited: the response can only
            // be framed by close. (Never fires for this app's replies,
            // which always write a body; the 417 path sets Connection.)
            keep_alive = false;
        }
        if !keep_alive {
            if !self.http10 && !connection_handler_set {
                connection = Some("close");
            }
        } else if self.http10 && !connection_handler_set {
            connection = Some("keep-alive");
        }
        let mut out = status_line.into_bytes();
        // Handler-set headers, sorted bytewise by key; EXTRA_ORDER keys
        // stay out of this block unless the handler set them.
        let mut sorted: Vec<&(String, String)> = self
            .headers
            .iter()
            .filter(|(k, _)| !EXTRA_ORDER.contains(&k.as_str()) || self.header_is_set(k))
            .collect();
        sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        for (k, v) in sorted {
            out.extend_from_slice(k.as_bytes());
            out.extend_from_slice(b": ");
            out.extend_from_slice(v.as_bytes());
            out.extend_from_slice(b"\r\n");
        }
        // Derived extras, EXTRA_ORDER order.
        if !self.header_is_set("Date") {
            out.extend_from_slice(b"Date: ");
            out.extend_from_slice(http_date(now_epoch_secs()).as_bytes());
            out.extend_from_slice(b"\r\n");
        }
        if !self.header_is_set("Content-Length")
            && (self.body.len() > 0 || (!self.is_head && keep_alive))
        {
            out.extend_from_slice(b"Content-Length: ");
            out.extend_from_slice(self.body.len().to_string().as_bytes());
            out.extend_from_slice(b"\r\n");
        }
        if let Some(value) = connection {
            out.extend_from_slice(b"Connection: ");
            out.extend_from_slice(value.as_bytes());
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"\r\n");
        if !self.is_head {
            out.extend_from_slice(&self.body);
        }
        (out, keep_alive)
    }
}

/// IMF-fixdate: "Mon, 02 Jan 2006 15:04:05 GMT".
fn http_date(secs: u64) -> String {
    const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let days = secs / 86400;
    let sod = secs % 86400;
    // Howard Hinnant's civil_from_days over days since 1970-01-01.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let weekday = ((days + 4) % 7) as usize; // 1970-01-01 was a Thursday.
    crate::format!(
        "{}, {:02} {} {:04} {:02}:{:02}:{:02} GMT",
        DAYS[weekday],
        d,
        MONTHS[(m - 1) as usize],
        y,
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Go's StatusText for the codes this stack emits.
fn status_text(code: u16) -> &'static str {
    match code {
        100 => "Continue",
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Request Entity Too Large",
        415 => "Unsupported Media Type",
        417 => "Expectation Failed",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "",
    }
}

/// Raw parse-level error bytes (goish __status_error_response shape:
/// status line + text/plain block + body).
fn plain_error(status_line_text: &str, body: &str) -> Vec<u8> {
    crate::format!(
        "HTTP/1.1 {t}\r\nContent-Type: text/plain; charset=utf-8\r\nConnection: close\r\n\r\n{b}",
        t = status_line_text,
        b = body
    )
    .into_bytes()
}

fn bad_request() -> Vec<u8> {
    plain_error("400 Bad Request", "400 Bad Request")
}

fn request_too_large() -> Vec<u8> {
    plain_error(
        "431 Request Header Fields Too Large",
        "431 Request Header Fields Too Large",
    )
}

fn unsupported_te() -> Vec<u8> {
    plain_error("501 Not Implemented", "Unsupported transfer encoding")
}

fn is_token(b: u8) -> bool {
    b.is_ascii_alphanumeric()
        || matches!(
            b,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

fn valid_method(m: &str) -> bool {
    !m.is_empty() && m.bytes().all(is_token)
}

fn valid_header_name(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(is_token)
}

/// Bytes through the end of the head (first empty line), LF-tolerant.
fn head_terminator(buf: &[u8]) -> Option<usize> {
    let mut start = 0;
    for raw in buf.split(|&b| b == b'\n') {
        let end = start + raw.len() + 1;
        let line = raw.strip_suffix(b"\r").unwrap_or(raw);
        if line.is_empty() {
            return Some(end);
        }
        start = end;
    }
    None
}

struct Head {
    method: String,
    path: String,
    http10: bool,
    headers: Vec<(String, String)>,
    chunked: bool,
    content_length: i64,
    keep_alive: bool,
}

enum Stage {
    Ok(Head),
    /// Respond with these bytes, then close.
    Reject(Vec<u8>),
    /// Header line over the per-line cap under the total cap: silent
    /// close at deadline (observed goish behavior).
    Silent,
    Incomplete,
}

fn parse_head(buf: &[u8]) -> Stage {
    let head_end = match head_terminator(buf) {
        Some(n) => n,
        None => {
            if buf.len() > MAX_HEAD {
                // Total-head cap (initialReadLimitSize): 431 + close.
                return Stage::Reject(request_too_large());
            }
            return Stage::Incomplete;
        }
    };
    if head_end > MAX_HEAD {
        return Stage::Reject(request_too_large());
    }
    let mut lines: Vec<&[u8]> = Vec::new();
    for raw in buf[..head_end].split(|&b| b == b'\n') {
        let line = raw.strip_suffix(b"\r").unwrap_or(raw);
        if line.is_empty() {
            break;
        }
        if line.len() > MAX_LINE {
            return Stage::Silent;
        }
        lines.push(line);
    }
    let (first, rest) = match lines.split_first() {
        Some(v) => v,
        None => return Stage::Reject(bad_request()),
    };
    let request_line = String::from_utf8_lossy(first).into_owned();
    // Exactly three space-separated tokens (Go parseRequestLine: a target
    // or version containing a space is "malformed HTTP request" -> 400).
    let mut parts = request_line.split(' ');
    let (method, target, version) = match (parts.next(), parts.next(), parts.next()) {
        (Some(m), Some(t), Some(v)) if parts.next().is_none() => {
            (m.to_string(), t.to_string(), v.to_string())
        }
        _ => return Stage::Reject(bad_request()),
    };
    if method.is_empty() || target.is_empty() || version.is_empty() {
        return Stage::Reject(bad_request());
    }
    if !valid_method(&method) {
        return Stage::Reject(bad_request());
    }
    let http10 = match version.trim_end() {
        "HTTP/1.1" => false,
        "HTTP/1.0" => true,
        _ => {
            return Stage::Reject(plain_error(
                "505 HTTP Version Not Supported: unsupported protocol version",
                "505 HTTP Version Not Supported: unsupported protocol version",
            ))
        }
    };
    // Absolute-form request targets (RFC 7230 §5.3.2) resolve to their
    // path; query strings never reach the handler.
    let target = match target
        .strip_prefix("http://")
        .or_else(|| target.strip_prefix("https://"))
    {
        Some(rest) => match rest.find('/') {
            Some(i) => &rest[i..],
            None => "/",
        },
        None => target.as_str(),
    };
    let path = match target.split_once('?') {
        Some((p, _)) => p.to_string(),
        None => target.to_string(),
    };
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut host_count = 0usize;
    for line in rest {
        let line = String::from_utf8_lossy(line).into_owned();
        let (name, value) = match line.split_once(':') {
            Some((n, v)) if !n.is_empty() => (n.to_string(), v.trim().to_string()),
            _ => return Stage::Reject(bad_request()),
        };
        if !valid_header_name(&name) {
            return Stage::Reject(bad_request());
        }
        if name.eq_ignore_ascii_case("host") {
            host_count += 1;
            if host_count > 1 {
                return Stage::Reject(bad_request());
            }
        }
        headers.push((canonical_key(&name), value));
    }
    let mut te: Option<String> = None;
    let mut te_count = 0usize;
    let mut cl: Option<Result<i64, ()>> = None;
    for (k, v) in &headers {
        if k == "Transfer-Encoding" {
            te_count += 1;
            te = Some(v.to_string());
        } else if k == "Content-Length" {
            let parsed = v.trim().parse::<i64>().map_err(|_| ()).and_then(|n| {
                if n < 0 {
                    Err(())
                } else {
                    Ok(n)
                }
            });
            match &cl {
                Some(Ok(prev)) if parsed.as_ref() == Ok(prev) => {}
                Some(_) => return Stage::Reject(bad_request()),
                None => cl = Some(parsed),
            }
        }
    }
    if te_count > 1 {
        return Stage::Reject(unsupported_te());
    }
    let chunked;
    let content_length;
    match te {
        Some(v) => {
            let coding = v.rsplit(',').next().unwrap_or("").trim();
            if coding.eq_ignore_ascii_case("chunked") {
                chunked = true;
                content_length = -1; // TE wins; any Content-Length is ignored.
            } else if coding.eq_ignore_ascii_case("identity") {
                chunked = false;
                content_length = cl.map(|r| r.unwrap_or(0)).unwrap_or(0);
            } else {
                return Stage::Reject(unsupported_te());
            }
        }
        None => {
            chunked = false;
            content_length = match cl {
                Some(Ok(n)) => n,
                Some(Err(())) => return Stage::Reject(bad_request()),
                None => 0,
            };
        }
    }
    if !chunked && content_length as usize > MAX_BODY {
        // Declared body over the 16 MiB parser cap: the client sees the
        // generic 400 (goish readRequest path).
        return Stage::Reject(bad_request());
    }
    let conn = headers
        .iter()
        .filter(|(k, _)| k == "Connection")
        .map(|(_, v)| v.as_str())
        .collect::<String>();
    let has_token = |t: &str| conn.split(',').any(|p| p.trim().eq_ignore_ascii_case(t));
    let keep_alive = if http10 {
        has_token("keep-alive")
    } else {
        !has_token("close")
    };
    Stage::Ok(Head {
        method,
        path,
        http10,
        headers,
        chunked,
        content_length,
        keep_alive,
    })
}

/// Chunked decoder; Some((decoded, consumed)) when the body plus trailer
/// section is fully buffered, None when more bytes are needed, Err on
/// malformed framing or over the 16 MiB cap.
fn decode_chunked(buf: &[u8]) -> Result<Option<(Vec<u8>, usize)>, ()> {
    // One LF-terminated line from `from`: (end, content-with-CRLF-stripped).
    fn line_at(buf: &[u8], from: usize) -> Option<(usize, &[u8])> {
        let n = buf[from..].iter().position(|&b| b == b'\n')?;
        let end = from + n + 1;
        let raw = &buf[from..end - 1];
        let line = raw.strip_suffix(b"\r").unwrap_or(raw);
        Some((end, line))
    }
    let mut decoded = Vec::new();
    let mut pos = 0;
    loop {
        let (line_end, line) = match line_at(buf, pos) {
            Some(v) => v,
            None => return Ok(None),
        };
        let size_str = line.split(|&b| b == b';').next().unwrap_or(line);
        let size = core::str::from_utf8(size_str)
            .ok()
            .and_then(|s| usize::from_str_radix(s.trim(), 16).ok())
            .ok_or(())?;
        if size == 0 {
            let mut p = line_end;
            loop {
                let (end, l) = match line_at(buf, p) {
                    Some(v) => v,
                    None => return Ok(None),
                };
                p = end;
                if l.is_empty() {
                    return Ok(Some((decoded, p)));
                }
            }
        }
        if decoded.len() + size > MAX_BODY {
            return Err(());
        }
        if buf.len() < line_end + size + 1 {
            return Ok(None);
        }
        decoded.extend_from_slice(&buf[line_end..line_end + size]);
        if buf[line_end + size] == b'\n' {
            pos = line_end + size + 1;
        } else if buf.len() >= line_end + size + 2
            && &buf[line_end + size..line_end + size + 2] == b"\r\n"
        {
            pos = line_end + size + 2;
        } else {
            return Err(());
        }
    }
}

/// Buffered connection reader with deadline, drain-aware: returns false
/// on EOF, deadline expiry, drain while still idle, or hard error.
fn read_more(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
    deadline: Instant,
    shared: &Shared,
) -> bool {
    let _ = stream.set_read_timeout(Some(POLL));
    let mut chunk = [0u8; 16384];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => return false,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                return true;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut => {
                if Instant::now() >= deadline {
                    return false;
                }
                if shared.draining.load(Ordering::Acquire) {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }
}

/// Client-disconnect probe used between encoder forwards: a zero-timeout
/// poll(2) for POLLIN|POLLRDHUP|POLLHUP|POLLERR on a dup of the socket
/// (goish checks the request context's Err() the same way between
/// forwards). A timeout-based peek here measured 22 ms per call on this
/// kernel (SO_RCVTIMEO slow path) and cost the serve path +14% latency;
/// poll(0) is sub-microsecond.
pub struct Cancel {
    peeker: Option<TcpStream>,
}

#[cfg(unix)]
mod cancel_probe {
    const POLLIN: i16 = 0x001;
    const POLLHUP: i16 = 0x004;
    const POLLERR: i16 = 0x008;
    const POLLRDHUP: i16 = 0x2000;
    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    extern "C" {
        fn poll(fds: *mut PollFd, nfds: u64, timeout: i32) -> i32;
        fn recv(fd: i32, buf: *mut u8, len: usize, flags: i32) -> isize;
    }
    const MSG_PEEK: i32 = 2;
    const MSG_DONTWAIT: i32 = 0x40;
    /// True when the peer has closed. Data still queued (pipelined
    /// request bytes) reads as data, not as a disconnect.
    pub fn peer_closed(fd: i32) -> bool {
        let mut pfd = PollFd {
            fd,
            events: POLLIN | POLLRDHUP,
            revents: 0,
        };
        let rc = unsafe { poll(&mut pfd, 1, 0) };
        if rc <= 0 {
            return false;
        }
        if pfd.revents & (POLLHUP | POLLERR | POLLRDHUP) != 0 {
            return true;
        }
        if pfd.revents & POLLIN != 0 {
            let mut byte = [0u8; 1];
            let n = unsafe { recv(fd, byte.as_mut_ptr(), 1, MSG_PEEK | MSG_DONTWAIT) };
            return n == 0;
        }
        false
    }
}

impl Cancel {
    pub fn watch(stream: &TcpStream) -> Self {
        Self {
            peeker: stream.try_clone().ok(),
        }
    }
    pub fn canceled(&self) -> bool {
        let Some(s) = self.peeker.as_ref() else {
            return false;
        };
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            cancel_probe::peer_closed(s.as_raw_fd())
        }
        #[cfg(not(unix))]
        {
            false
        }
    }
}

enum Outcome {
    /// Handled; write these bytes. bool = keep the connection open.
    Reply(Vec<u8>, bool),
    /// Parse-level reject: write then close.
    Reject(Vec<u8>),
    /// Close silently.
    Silent,
}

/// Read one request off the buffer (reading more from the stream as
/// needed), then dispatch to the handler. Mirrors goish's per-request
/// sequence: skip-CRLF-after-POST, head parse, 100-continue / 417,
/// eager body read, handler, rendered response.
fn serve_one<H, F>(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
    first_request: bool,
    last_method_post: bool,
    handler: &H,
    dispatch: &F,
    shared: &Shared,
) -> Outcome
where
    F: Fn(&H, &mut Response, &Request, &Cancel),
{
    // RFC 7230 §3 tolerance: skip leading CRLF after a POST body.
    if last_method_post {
        let n = buf
            .iter()
            .take_while(|&&b| b == b'\r' || b == b'\n')
            .count();
        buf.drain(..n);
    }
    let deadline = Instant::now()
        + if first_request {
            READ_HEADER_TIMEOUT
        } else {
            IDLE_TIMEOUT
        };
    loop {
        match parse_head(buf) {
            Stage::Reject(bytes) => return Outcome::Reject(bytes),
            Stage::Silent => {
                // Keep the connection open until the request deadline
                // (observed: goish closes at the 5 s header deadline).
                while Instant::now() < deadline {
                    std::thread::sleep(POLL.min(deadline - Instant::now()));
                    if shared.draining.load(Ordering::Acquire) {
                        break;
                    }
                }
                return Outcome::Silent;
            }
            Stage::Incomplete => {
                if !read_more(stream, buf, deadline, shared) {
                    return Outcome::Silent;
                }
            }
            Stage::Ok(head) => {
                let head_end = head_terminator(buf).unwrap_or(0);
                let body_indicated = head.chunked || head.content_length > 0;
                let expect = head
                    .headers
                    .iter()
                    .find(|(k, _)| k == "Expect")
                    .map(|(_, v)| v.trim().to_string())
                    .unwrap_or_default();
                if body_indicated && !head.http10 && !expect.is_empty() {
                    if expect.eq_ignore_ascii_case("100-continue") {
                        let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
                    } else {
                        // sendExpectationFailed: handler-set Connection:
                        // close sorts before Date; zero-body explicit
                        // WriteHeader advertises Content-Length: 0.
                        let mut out = b"HTTP/1.1 417 Expectation Failed\r\n".to_vec();
                        out.extend_from_slice(b"Connection: close\r\n");
                        out.extend_from_slice(b"Date: ");
                        out.extend_from_slice(http_date(now_epoch_secs()).as_bytes());
                        out.extend_from_slice(b"\r\nContent-Length: 0\r\n\r\n");
                        return Outcome::Reject(out);
                    }
                }
                // Eager body read (16 MiB parser cap enforced here).
                let (body, consumed) = if head.chunked {
                    match decode_chunked(&buf[head_end..]) {
                        Ok(Some((b, c))) => (b, head_end + c),
                        Ok(None) => {
                            if read_more(stream, buf, deadline, shared) {
                                continue;
                            }
                            return Outcome::Silent;
                        }
                        Err(()) => return Outcome::Reject(bad_request()),
                    }
                } else {
                    let n = head.content_length as usize;
                    if buf.len() < head_end + n {
                        if read_more(stream, buf, deadline, shared) {
                            continue;
                        }
                        return Outcome::Silent;
                    }
                    (buf[head_end..head_end + n].to_vec(), head_end + n)
                };
                let was_post = head.method == "POST";
                let request = Request {
                    method: head.method,
                    path: head.path,
                    headers: head.headers,
                    content_length: head.content_length,
                    body,
                    http10: head.http10,
                    keep_alive: head.keep_alive,
                };
                let mut response = Response::new(
                    &request.method,
                    request.http10,
                    request.keep_alive && !shared.draining.load(Ordering::Acquire),
                );
                let cancel = Cancel::watch(stream);
                dispatch(handler, &mut response, &request, &cancel);
                let (bytes, keep) = response.render();
                buf.drain(..consumed);
                LAST_POST.with(|c| c.set(was_post));
                return Outcome::Reply(bytes, keep);
            }
        }
    }
}

thread_local! {
    static LAST_POST: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

fn conn_thread<H, F>(mut stream: TcpStream, handler: Arc<H>, dispatch: Arc<F>, shared: Arc<Shared>)
where
    H: Send + Sync + 'static,
    F: Fn(&H, &mut Response, &Request, &Cancel) + Send + Sync + 'static,
{
    let _ = stream.set_nodelay(true);
    let mut buf: Vec<u8> = Vec::new();
    let mut first = true;
    loop {
        if buf.is_empty() {
            // Idle wait (also the pre-first-request wait): drain closes
            // idle keep-alive connections promptly.
            if shared.draining.load(Ordering::Acquire) {
                return;
            }
            let deadline = Instant::now()
                + if first {
                    READ_HEADER_TIMEOUT
                } else {
                    IDLE_TIMEOUT
                };
            if !read_more(&mut stream, &mut buf, deadline, &shared) {
                return;
            }
        }
        shared.enter();
        let last_post = LAST_POST.with(|c| c.get());
        let outcome = serve_one(
            &mut stream,
            &mut buf,
            first,
            last_post,
            handler.as_ref(),
            dispatch.as_ref(),
            shared.as_ref(),
        );
        shared.leave();
        match outcome {
            Outcome::Reject(bytes) => {
                let _ = stream.write_all(&bytes);
                return;
            }
            Outcome::Silent => return,
            Outcome::Reply(bytes, keep) => {
                if stream.write_all(&bytes).is_err() {
                    return;
                }
                if !keep || shared.draining.load(Ordering::Acquire) {
                    return;
                }
            }
        }
        first = false;
    }
}

/// Bind, announce, accept until drain; wait for in-flight requests
/// (30 s); Err when the deadline was exceeded. `on_drain` runs when the
/// signal arrives, after `draining` is set (the reference logs its drain
/// line at that point).
pub fn serve<H, F>(
    addr: &SocketAddr,
    shared: Arc<Shared>,
    handler: Arc<H>,
    dispatch: F,
    announce: &dyn Fn(),
    on_drain: impl Fn() + Send + Sync + 'static,
) -> Result<(), String>
where
    H: Send + Sync + 'static,
    F: Fn(&H, &mut Response, &Request, &Cancel) + Send + Sync + 'static,
{
    let listener = TcpListener::bind(addr).map_err(|e| crate::format!("listen {addr}: {e:?}"))?;
    // Non-blocking accept so a signal with no traffic in flight is still
    // observed promptly (the drain deadline starts at the signal).
    listener.set_nonblocking(true).ok();
    let drain_shared = shared.clone();
    install_signal_drain(move || {
        drain_shared.draining.store(true, Ordering::Release);
        on_drain();
    });
    announce();
    let dispatch = Arc::new(dispatch);
    let mut kids: Vec<std::thread::JoinHandle<()>> = Vec::new();
    loop {
        if shared.draining.load(Ordering::Acquire) {
            break;
        }
        // MaxConcurrentConns: a connection holds a slot for its whole
        // life; at 32 live connections the loop stops accepting and the
        // kernel backlog holds newcomers (goish semaphore semantics).
        // The sweep also joins finished threads — an unjoined JoinHandle
        // keeps the thread's 2 MiB stack mapped for the process lifetime,
        // which was the +300 MiB soak plateau (wayfinder ticket 21).
        while sweep_conns(&mut kids) >= MAX_CONNS && !shared.draining.load(Ordering::Acquire) {
            std::thread::sleep(POLL);
        }
        if shared.draining.load(Ordering::Acquire) {
            break;
        }
        match listener.accept() {
            Ok((stream, _)) => {
                let handler = handler.clone();
                let dispatch = dispatch.clone();
                let shared = shared.clone();
                if let Ok(kid) = std::thread::Builder::new()
                    .name("laya-conn".into())
                    .spawn(move || conn_thread(stream, handler, dispatch, shared))
                {
                    kids.push(kid);
                }
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                if shared.draining.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(ACCEPT_POLL);
            }
            Err(_) => {
                if shared.draining.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(ACCEPT_POLL);
            }
        }
    }
    drop(listener);
    let deadline = Instant::now() + DRAIN_TIMEOUT;
    while shared.active() > 0 && Instant::now() < deadline {
        std::thread::sleep(POLL);
    }
    if shared.active() > 0 {
        return Err("HTTP shutdown deadline exceeded".into());
    }
    Ok(())
}

fn sweep_conns(kids: &mut Vec<std::thread::JoinHandle<()>>) -> usize {
    let mut live = 0;
    let mut i = 0;
    while i < kids.len() {
        if kids[i].is_finished() {
            let kid = kids.swap_remove(i);
            let _ = kid.join();
        } else {
            live += 1;
            i += 1;
        }
    }
    live
}

// Signal plumbing: a self-pipe written by the raw signal handler (one
// write syscall, async-signal-safe) and drained by a monitor thread that
// flips `draining` — no locks in signal context.
#[cfg(unix)]
static PIPE_W: AtomicI32 = AtomicI32::new(-1);

#[cfg(unix)]
extern "C" fn on_signal(_sig: i32) {
    let fd = PIPE_W.load(Ordering::Relaxed);
    if fd >= 0 {
        extern "C" {
            fn write(fd: i32, buf: *const u8, count: usize) -> isize;
        }
        let byte = 1u8;
        unsafe {
            let _ = write(fd, &byte as *const u8, 1);
        }
    }
}

#[cfg(unix)]
fn install_signal_drain<F: Fn() + Send + Sync + 'static>(on_drain: F) {
    extern "C" {
        fn pipe(fds: *mut i32) -> i32;
        fn signal(signum: i32, handler: usize) -> usize;
    }
    let mut fds = [0i32; 2];
    if unsafe { pipe(fds.as_mut_ptr()) } != 0 {
        return;
    }
    PIPE_W.store(fds[1], Ordering::Release);
    unsafe {
        signal(2, on_signal as *const () as usize); // SIGINT
        signal(15, on_signal as *const () as usize); // SIGTERM
    }
    let read_fd = fds[0];
    std::thread::Builder::new()
        .name("laya-drain".into())
        .spawn(move || {
            extern "C" {
                fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
            }
            let mut byte = [0u8; 1];
            loop {
                let n = unsafe { read(read_fd, byte.as_mut_ptr(), 1) };
                if n > 0 {
                    on_drain();
                    return;
                }
                if n == 0 {
                    return;
                }
            }
        })
        .ok();
}

// Windows capture arm (ticket 23): SetConsoleCtrlHandler against
// kernel32 via extern declarations only (std already links kernel32 on
// both windows-msvc and windows-gnu — no new crate). The OS runs the
// handler on its own console-ctrl thread, but it stays as minimal as
// the unix raw handler: one atomic store, nothing else. A "laya-drain"
// monitor thread — the same role as the unix self-pipe reader — polls
// that flag at accept-loop granularity and runs the identical on_drain
// closure; everything below the capture seam is shared code.
#[cfg(windows)]
static CTRL_SEEN: AtomicBool = AtomicBool::new(false);

#[cfg(windows)]
extern "system" fn on_ctrl_event(_ctrl_type: u32) -> i32 {
    CTRL_SEEN.store(true, Ordering::Release);
    1 // TRUE: handled — the default ^C/^Break termination is suppressed.
}

#[cfg(windows)]
fn install_signal_drain<F: Fn() + Send + Sync + 'static>(on_drain: F) {
    extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<extern "system" fn(u32) -> i32>, add: i32) -> i32;
    }
    if unsafe { SetConsoleCtrlHandler(Some(on_ctrl_event), 1) } == 0 {
        return; // failure means no console is attached: no ctrl events
    } // will ever arrive
    std::thread::Builder::new()
        .name("laya-drain".into())
        .spawn(move || loop {
            if CTRL_SEEN.load(Ordering::Acquire) {
                on_drain();
                return;
            }
            std::thread::sleep(ACCEPT_POLL);
        })
        .ok();
}

// wasm32-unknown-unknown has neither signals nor a drain thread (ticket 19
// wasm tier: the serve path is native-only; the wasm ABI drives forward
// directly).
#[cfg(not(any(unix, windows)))]
fn install_signal_drain<F: Fn() + Send + Sync + 'static>(on_drain: F) {
    let _ = on_drain;
}
