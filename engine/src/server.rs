//! HTTP application layer: the decision API surface (routes, auth,
//! registry routing, busy serialization, batch, drain), ported from the
//! frozen goish reference src/server.rs. Transport lives in http.rs;
//! this file must mirror the reference's behavior exactly.
use crate::{
    http::{self, Cancel, Request, Response},
    io,
    json::{self, J},
    registry::Registry,
    request,
};
use alloc::{string::String, sync::Arc, vec, vec::Vec};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::net::ToSocketAddrs;

const MAX_BODY: i64 = 1024 * 1024;

struct App {
    registry: Registry,
    presets: J,
    key: String,
    raw: bool,
    busy: AtomicBool,
    shared: Arc<http::Shared>,
    requests: AtomicU64,
}

/// Releases the busy flag when dropped, mirroring the reference's Busy
/// guard (inference owns the flag from CAS to response).
struct Busy<'a>(&'a AtomicBool);
impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn reply(w: &mut Response, code: u16, body: J) {
    w.set_header("Content-Type", "application/json; charset=utf-8");
    w.set_header("Cache-Control", "no-store");
    w.set_header("X-Content-Type-Options", "nosniff");
    w.write_header(code);
    w.write(crate::format!("{}\n", body.dump()).as_bytes());
}

fn error(w: &mut Response, code: u16, kind: &str, message: &str) {
    reply(
        w,
        code,
        J::object(vec![(
            "error",
            J::object(vec![("type", J::s(kind)), ("message", J::s(message))]),
        )]),
    );
}

/// Length check first, then a constant-time compare — exactly the
/// reference's secret_equal.
fn secret_equal(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |v, (a, b)| v | (a ^ b)) == 0
}

impl App {
    fn validate(&self, model: &crate::engine::Engine, req: &J) -> Result<J, String> {
        if !matches!(req, J::Obj(_)) {
            return Err("request body must be a JSON object".into());
        }
        if req.get("state") == &J::Null && req.get("preset").str().is_empty() {
            return Err("state is required (or supply a preset)".into());
        }
        let resolved = request::resolve(req)?;
        model.validate(&resolved)?;
        Ok(resolved)
    }

    fn handle(&self, w: &mut Response, r: &Request, cancel: &Cancel) {
        let request_id = self.requests.fetch_add(1, Ordering::Relaxed) + 1;
        w.set_header("X-Request-Id", &crate::format!("laya-{request_id}"));
        let path = r.path.as_str();
        let method = r.method.as_str();
        let allowed = match path {
            "/" | "/health" | "/v1/models" | "/v1/presets" => "GET, HEAD",
            "/v1/systemone" | "/v1/decide" | "/v1/decide/batch" => "POST",
            _ => {
                error(w, 404, "not_found_error", "Endpoint not found");
                return;
            }
        };
        if (method != "GET" && method != "HEAD" && allowed == "GET, HEAD")
            || (method != "POST" && allowed == "POST")
        {
            w.set_header("Allow", allowed);
            error(w, 405, "invalid_request_error", "Method not allowed");
            return;
        }
        let public = path == "/" || path == "/health" || path == "/v1/presets";
        if !public && !self.key.is_empty() {
            let auth = r.header("Authorization");
            if !auth
                .strip_prefix("Bearer ")
                .map(|s| secret_equal(s.as_bytes(), self.key.as_bytes()))
                .unwrap_or(false)
            {
                w.set_header("WWW-Authenticate", "Bearer");
                error(
                    w,
                    401,
                    "authentication_error",
                    "Invalid or missing bearer token",
                );
                return;
            }
        }
        match path {
            "/" => {
                reply(
                    w,
                    200,
                    J::object(vec![
                        ("service", J::s("Laya / Goish Rust")),
                        ("health", J::s("/health")),
                        ("decide", J::s("/v1/systemone")),
                        ("presets", J::s("/v1/presets")),
                    ]),
                );
                return;
            }
            "/health" => {
                let draining = self.shared.draining.load(Ordering::Acquire);
                reply(
                    w,
                    if draining { 503 } else { 200 },
                    J::object(vec![
                        ("status", J::s(if draining { "draining" } else { "ok" })),
                        ("model", J::s(self.registry.default_model().name())),
                        (
                            "family",
                            J::s(self.registry.default_model().info.family.as_str()),
                        ),
                        ("device", J::s("cpu")),
                        ("models", self.registry.loaded_names()),
                        ("busy", J::Bool(self.busy.load(Ordering::Acquire))),
                    ]),
                );
                return;
            }
            "/v1/models" => {
                reply(w, 200, self.registry.listing());
                return;
            }
            "/v1/presets" => {
                reply(
                    w,
                    200,
                    J::Arr(
                        self.presets
                            .obj()
                            .iter()
                            .map(|(n, p)| {
                                let mut fields = p.obj().to_vec();
                                fields.insert(0, ("name".into(), J::s(n.clone())));
                                J::Obj(fields)
                            })
                            .collect(),
                    ),
                );
                return;
            }
            _ => {}
        }
        if self.shared.draining.load(Ordering::Acquire) {
            error(w, 503, "unavailable_error", "Server is shutting down");
            return;
        }
        let content_type = r.header("Content-Type");
        if !content_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .eq_ignore_ascii_case("application/json")
        {
            error(
                w,
                415,
                "invalid_request_error",
                "Content-Type must be application/json",
            );
            return;
        }
        // The transport buffers the body (16 MiB parser cap) before this
        // point; the app limit is enforced for both fixed-length and
        // chunked input exactly like the reference.
        if r.content_length > MAX_BODY {
            error(
                w,
                413,
                "invalid_request_error",
                "Request body exceeds 1 MiB",
            );
            return;
        }
        if r.body.len() > MAX_BODY as usize {
            error(
                w,
                413,
                "invalid_request_error",
                "Request body exceeds 1 MiB or could not be read",
            );
            return;
        }
        let text = match core::str::from_utf8(&r.body) {
            Ok(v) => v,
            Err(_) => {
                error(w, 400, "invalid_request_error", "Request body is not UTF-8");
                return;
            }
        };
        let req = match json::parse(text) {
            Ok(v) => v,
            Err(e) => {
                error(w, 400, "invalid_request_error", &e);
                return;
            }
        };
        let model = match self.registry.select(&req) {
            Ok(model) => model,
            Err(e) => {
                error(w, 422, "invalid_request_error", &e);
                return;
            }
        };
        let batch = path == "/v1/decide/batch";
        let mut requests = Vec::new();
        if batch {
            if req.get("states").arr().is_empty() || req.get("states").arr().len() > 256 {
                error(
                    w,
                    422,
                    "invalid_request_error",
                    "states must be an array of 1..256 states",
                );
                return;
            }
            for state in req.get("states").arr() {
                requests.push(J::object(vec![
                    ("state", state.clone()),
                    ("questions", req.get("questions").clone()),
                    ("model", req.get("model").clone()),
                    ("order_invariant", req.get("order_invariant").clone()),
                    ("permutations", req.get("permutations").clone()),
                ]));
            }
        } else {
            requests.push(req);
        }
        let mut resolved = Vec::new();
        let mut question_count = 0;
        // Validate the whole batch before starting any expensive work.
        for req in &requests {
            match self.validate(model, req) {
                Ok(v) => {
                    question_count += v.get("questions").obj().len();
                    resolved.push(v);
                }
                Err(e) => {
                    error(w, 422, "invalid_request_error", &e);
                    return;
                }
            }
        }
        if question_count > 256 {
            error(
                w,
                422,
                "invalid_request_error",
                "At most 256 total questions per request",
            );
            return;
        }
        // One inference at a time across all loaded models (team.rs
        // assumes a single coordinator thread; the busy flag provides
        // it, exactly like the goish server).
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            w.set_header("Retry-After", "1");
            error(
                w,
                429,
                "busy_error",
                "An inference request is running; retry later",
            );
            return;
        }
        let _busy = Busy(&self.busy);
        let start = std::time::Instant::now();
        let mut results = Vec::new();
        for req in &resolved {
            match model.decide(req, self.raw, &|| cancel.canceled()) {
                Ok(v) => results.push(v),
                Err(e) => {
                    let canceled = cancel.canceled();
                    error(
                        w,
                        if canceled { 408 } else { 500 },
                        if canceled {
                            "request_canceled"
                        } else {
                            "internal_error"
                        },
                        &e,
                    );
                    return;
                }
            }
        }
        w.set_header(
            "X-Response-Time-Ms",
            &crate::format!("{}", start.elapsed().as_millis()),
        );
        reply(
            w,
            200,
            if batch {
                J::object(vec![("results", J::Arr(results))])
            } else {
                results.remove(0)
            },
        );
    }
}

pub fn serve(registry: Registry, host: &str, port: u16, raw: bool) -> Result<(), String> {
    let mut key = std::env::var("LAYA_API_KEY").unwrap_or_default();
    if key.is_empty() {
        key = std::env::var("TYPESAFE_API_KEY").unwrap_or_default();
    }
    let auth_enabled = !key.is_empty();
    let shared = Arc::new(http::Shared::new());
    let app = Arc::new(App {
        registry,
        presets: json::parse(include_str!("../presets.json"))?,
        key,
        raw,
        busy: AtomicBool::new(false),
        shared: shared.clone(),
        requests: AtomicU64::new(0),
    });
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|e| crate::format!("listen {host}:{port}: {e:?}"))?
        .next()
        .ok_or_else(|| crate::format!("listen {host}:{port}: no addresses"))?;
    let announce_line = crate::format!(
        "laya HTTP listening at http://{addr} (bearer auth: {})",
        if auth_enabled { "on" } else { "off" }
    );
    let announce = || io::log(&announce_line);
    http::serve(
        &addr,
        shared,
        app,
        |app: &App, w: &mut Response, r: &Request, cancel: &Cancel| app.handle(w, r, cancel),
        &announce,
        || io::log("laya HTTP: draining requests (30 second deadline)"),
    )
}
