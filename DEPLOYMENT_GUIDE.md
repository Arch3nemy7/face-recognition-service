# Deployment Guide

This guide covers deploying the face recognition service with the
`docker-compose.yml` this repo actually ships: a bundled `nginx` reverse
proxy plus a `certbot` container using Cloudflare's DNS-01 challenge,
terminating TLS on the box itself (no external reverse-proxy stack is
assumed). `face-recognition` joins an internal-only network
(`INTERNAL_NETWORK_NAME`); `nginx` also joins an external network
(`EXTERNAL_NETWORK_NAME`) and publishes ports 80/443.

If you're looking for the API contract (endpoints, request/response shapes,
error codes), see [README.md](README.md). This guide is deploy/operate only.

## 1. Prerequisites

- Docker and Docker Compose (`docker compose`, the plugin form) on the VPS.
- A domain name with Cloudflare as its DNS provider, pointed at the VPS.
- A Cloudflare API token with `Zone.Zone:Read` and `Zone.DNS:Edit`
  permissions, scoped to that domain.
- SSH access to the VPS.

## 2. Configuration

`docker-compose.yml` reads a `.env` file next to it for variable
substitution (`${VAR:-default}` / `${VAR:?error}`). Copy the template and
edit it:

```bash
cp .env.example .env
```

`.env.example` is the source of truth for every setting the service or
compose file reads, grouped with defaults and warnings; `tests/test_config_drift.py`
keeps it in sync with `face_recognition_service/config.py`. Below is only
what changes for production.

### Required

- **`API_TOKEN`** -- compose fails closed without it
  (`API_TOKEN=${API_TOKEN:?API_TOKEN must be set}`), so every
  `docker compose` invocation -- `up`, `config`, `build`, all of them --
  needs `API_TOKEN` in the environment or in this `.env`. Generate one:
  ```bash
  openssl rand -hex 32
  ```
  This is the bearer token the calling application presents on every
  request; rotating it means updating both sides together.
- **`DOMAIN_NAME`** -- your actual domain. Also replace every
  `{{DOMAIN_NAME}}` placeholder in `nginx/nginx.conf` by hand with the same
  value (nginx does not read `.env`; see §4 below).
- **`LETSENCRYPT_EMAIL`** -- contact address for Let's Encrypt certificate
  notifications. Nothing in `docker-compose.yml` or the running containers
  reads this variable automatically; it's used as the `--email` value you
  pass by hand to the one-shot certificate-issuance command in §4.
- **`CLOUDFLARE_API_TOKEN`** -- copy `cloudflare.ini.example` to
  `cloudflare.ini` (kept out of git; `chmod 600` it) and paste the same
  token in as `dns_cloudflare_api_token`. Certbot reads `cloudflare.ini`
  directly, not this environment variable, for the actual DNS-01 challenge
  -- this `.env` copy of the token exists only so you have one place to
  keep it alongside the rest of your production values; nothing consumes
  it from the environment.

### Recommended production values -- starting points to re-measure

`docker-compose.yml` and `.env.example` default `DETECTION_THRESHOLD` and
`MIN_FACE_QUALITY` to `0.1` rather than `config.py`'s conservative library
defaults (`0.5`/`0.7`). These are **starting points carried over from this
project's hardening work, not values proven on this deployment's own
data.** Before trusting them (or tightening back toward the library
defaults) in production, run the evaluation harness against your own
traffic -- see [docs/evaluation.md](docs/evaluation.md).

| Variable | Shipped default | Note |
|---|---|---|
| `DETECTION_THRESHOLD` | `0.1` | Differs from `config.py`'s own conservative library default (`0.5`, used when the service runs outside this compose file/`.env`). Re-measure per `docs/evaluation.md` before relying on it. |
| `MIN_FACE_QUALITY` | `0.1` | Same split and same caveat as `DETECTION_THRESHOLD` (library default `0.7`). |
| `COSINE_MATCH_THRESHOLD` | `0.5` | See README "Threshold guidance" -- calibrate against your own data, not a public dataset alone. |
| `ENHANCE_MODE` | `detect_fallback` | Detects on the original first, falls back to an enhanced copy only if nothing is found; always embeds from the original pixels. A reasonable default, not a measured conclusion for your deployment. |
| `MODEL_NAME` | `antelopev2` | The only pack the Dockerfile bakes into the image (see Dockerfile). Every threshold above is a starting point calibrated for this pack; switching packs needs its own re-run of `python -m evaluation` and its own baked model, or the container falls back to downloading the new pack at runtime -- see "Model pack" below. |
| Quality gates (`MIN_FACE_SIZE_PX`, `MIN_INTEROCULAR_PX`, `MIN_BLUR_VARIANCE`, `MAX_ABS_YAW_PROXY`, `MAX_ABS_ROLL_DEG`, `MIN_EMBEDDING_NORM`) | all `0` (off) | Off by design -- see the warning in `.env.example`. Enabling one without checking real quality-metric distributions first can reject photos your deployment already accepts today. |
| `MULTI_FACE_POLICY` | `largest` | The calling application may only understand a fixed set of 400 codes; `reject_ambiguous` emits `MULTIPLE_FACES_DETECTED`, which it won't recognize unless updated first. |

**Why the split between `config.py` and the deployment files**: `Settings`'
own code defaults (`DETECTION_THRESHOLD=0.5`, `MIN_FACE_QUALITY=0.7`) are
deliberately conservative for anyone running this package outside a tuned
deployment, with no evaluation data yet. `docker-compose.yml` and
`.env.example` override both to `0.1` as a starting point pending your own
measurement. If you ever run the service some other way (bare `uvicorn`, a
different compose file, a `.env` that doesn't set these two keys), you get
the conservative `0.5`/`0.7` pair instead -- verify your actual running
config: `docker compose exec face-recognition env | grep -E 'DETECTION_THRESHOLD|MIN_FACE_QUALITY'`.

### Model pack

The Dockerfile bakes exactly `antelopev2` into the image at build time. If
you deliberately want to run a different pack (`buffalo_l`, `buffalo_sc`,
or a private pack), either rebuild the image with a Dockerfile that bakes
that pack instead, or bind-mount a directory containing it over
`/app/.insightface` in `docker-compose.yml`, e.g.:

```yaml
    volumes:
      - ./my-model-packs:/app/.insightface
```

Do not use a *named* Docker volume for this: a named volume mounted over a
path with content in the image only copies that content in on first
container creation, so a later `docker compose down -v` or volume prune
silently empties it, forcing a from-network re-download (or a hard failure
with no network) and running an unpinned pack with no version marker. A
bind mount you control has none of that ambiguity. Whatever pack you run,
re-run `python -m evaluation` against it before trusting any threshold in
the table above.

## 3. Reference-URL hardening

The service fetches the reference photo (`image1` on `/compare-photos`)
directly from a URL the calling application supplies. Two settings bound
where that fetch is allowed to go (full mechanism in README "Reference URL
host/IP policy"):

- **`REFERENCE_URL_ALLOWED_HOSTS`** -- JSON array, exact case-insensitive
  host match, applied to the original URL and every redirect target. Empty
  (the default) allows any host.
- **`REFERENCE_URL_ALLOW_PRIVATE`** -- whether a reference URL may resolve
  to a private/loopback/CGNAT/ULA address. Defaults to `true` so a
  not-yet-configured deployment doesn't lock everyone out.

**Procedure before locking this down:**

1. Find the distinct hosts the calling application's reference-photo URLs
   actually use in production -- do not guess.
2. Set `REFERENCE_URL_ALLOWED_HOSTS=["photos.example.com"]` (your real
   host(s), as a literal JSON array string, same format as `CORS_ORIGINS`).
3. Confirm none of those hosts resolve to a private/CGNAT/ULA address (if
   one does, `REFERENCE_URL_ALLOW_PRIVATE=false` would break it).
4. Set `REFERENCE_URL_ALLOW_PRIVATE=false`.

Link-local/multicast/unspecified/reserved addresses (including the
cloud-metadata class, `169.254.169.254`) are always blocked, regardless of
either setting.

**Queue settings**, also relevant here because they bound the same request
path: `MAX_CONCURRENT_INFERENCE` (default `1`) and `INFERENCE_QUEUE_TIMEOUT`
(default `20.0`s) -- see "Capacity tuning" below.

## 4. TLS / certbot setup

1. Replace every `{{DOMAIN_NAME}}` placeholder in `nginx/nginx.conf` with
   your real domain (nginx does not read `.env`, so this is a manual edit,
   not a variable substitution):
   ```bash
   sed -i 's/{{DOMAIN_NAME}}/your-domain.com/g' nginx/nginx.conf
   ```
2. Create the external Docker network `nginx` joins, if it doesn't already
   exist:
   ```bash
   docker network create ${EXTERNAL_NETWORK_NAME:-proxy-network}
   ```
3. Copy `cloudflare.ini.example` to `cloudflare.ini`, fill in your
   Cloudflare API token, and lock its permissions down:
   ```bash
   cp cloudflare.ini.example cloudflare.ini
   chmod 600 cloudflare.ini
   ```
   `cloudflare.ini` must never be committed -- it holds a live credential.
4. **Issue the first certificate before bringing `nginx` up.** The
   `certbot` service's own entrypoint only *renews* an already-issued
   certificate (`certbot renew`, looped every 12 hours) -- against the
   empty `./nginx/ssl` a fresh clone starts with, `renew` finds nothing to
   renew and does nothing. `nginx`'s `ssl_certificate`/`ssl_certificate_key`
   paths won't exist yet either way, so if you bring the full stack up
   first, `nginx` will fail to start (and keep crash-looping under its
   `restart:` policy) until a certificate actually exists. Issue it
   one-shot, with `docker compose run` overriding the service's default
   entrypoint to invoke `certbot certonly` directly instead:
   ```bash
   docker compose run --rm --entrypoint certbot certbot \
     certonly --dns-cloudflare \
     --dns-cloudflare-credentials /cloudflare.ini \
     -d "$DOMAIN_NAME" \
     --email "$LETSENCRYPT_EMAIL" \
     --agree-tos -n
   ```
   (export `DOMAIN_NAME` and `LETSENCRYPT_EMAIL` from your `.env` first,
   e.g. `set -a; source .env; set +a`, or substitute the literal values
   directly in the command above.) This uses the same `./nginx/ssl` and
   `cloudflare.ini` mounts already declared on the `certbot` service in
   `docker-compose.yml` -- `docker compose run` only overrides the
   entrypoint/command, not the volumes/networks. Certbot creates a DNS TXT
   record via the Cloudflare API, waits for propagation (tens of seconds,
   typically), and Let's Encrypt validates it -- usually done within a few
   minutes. The certificate lands under `./nginx/ssl/live/<domain>/`,
   which `nginx`'s config already points at.
5. *Now* bring the full stack up (see "Deploy" below). `certbot`'s normal
   entrypoint takes over from here, looping `certbot renew` every 12 hours
   -- a no-op renewal attempt until the certificate is within its renewal
   window, then automatic from there. No cron job is needed for renewal
   itself, but nothing in this stack reloads `nginx` after a renewal
   replaces the certificate files on disk -- `nginx` will keep serving the
   old certificate from memory until it's restarted. After any renewal
   (check `docker compose logs certbot` for "Congratulations" or
   "renewed"), reload it manually:
   ```bash
   docker compose exec nginx nginx -s reload
   ```
   A 90-day Let's Encrypt certificate renews roughly every 60 days, so a
   periodic manual check (or your own external cron hitting the command
   above) is enough in practice; this repo doesn't wire up anything more
   automatic than that, to avoid giving the `certbot` container access to
   the Docker socket it would need to restart `nginx` itself.

`nginx`'s `server_name {{DOMAIN_NAME}}` line will fail to match anything
useful until step 1 is done, so complete the domain substitution before
issuing the first certificate.

## 5. Deploy / update / rollback

### Deploy

```bash
API_TOKEN=<your-token> docker compose build face-recognition
API_TOKEN=<your-token> docker compose up -d
```

(or put `API_TOKEN` and the rest of your production values in `.env` next
to `docker-compose.yml`, and drop the inline `API_TOKEN=...` prefix -- both
work, since compose reads `.env` automatically.)

### Health

This compose file publishes 80/443 through `nginx`, which proxies
`/api/v1/health` without requiring the bearer token (see
`nginx/nginx.conf`'s `location /api/v1/health` block and the app's own
`health_check` route, neither of which enforce auth on this path). A
command that actually works against this topology:

```bash
curl -k https://your-domain.com/api/v1/health
```

or, to check the app container directly without going through nginx/DNS:

```bash
docker compose exec face-recognition python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/api/v1/health').read())"
```

Either way, expect `200` with `"model_loaded": true` once the antelopev2
pack (baked into the image at build time) has finished loading. Until then
it returns `503` -- this is expected during the container's `start_period`
(60s in the compose healthcheck) right after a fresh start, not a failure
by itself. Watch `docker compose ps` for the `face-recognition` container
to report `healthy`.

### Update

```bash
git pull
API_TOKEN=<your-token> docker compose build face-recognition
API_TOKEN=<your-token> docker compose up -d
```

### Rollback

Redeploy the previous image or git tag the same way (`git checkout
<previous-tag>` then rebuild, or retag/re-pull a previously built image if
you push to a registry). There is no in-place config rollback needed for
most settings -- they're read fresh from `.env`/compose on each start.

One exception: if an `ENHANCE_MODE=detect_fallback` rollout needs
reverting, set `ENHANCE_MODE=always` in `.env` to restore always-on
enhancement. Do **not** try to do this via `ENHANCE_IMAGE` --
`docker-compose.yml`'s `environment:` list doesn't include `ENHANCE_IMAGE`,
so setting it in a host `.env` is silently ignored under this compose file;
`ENHANCE_MODE` is the only variable that reaches the container for this.

## 6. Observability

- **`X-Request-ID`**: every response carries one, either echoed from a
  valid inbound header (`^[A-Za-z0-9._-]{1,64}$`) or freshly generated
  (`uuid4().hex`). Every log line for that request, including work done off
  the event loop in a threadpool, carries it as `[request_id]` -- grep a
  support ticket's header value straight out of the logs.
- **Summary log line**: `/compare-photos`, `/compare-photos-upload` and
  `/embed` each emit exactly one line per request, at INFO, on the
  `face_recognition_service.summary` logger, as flat `key=value` pairs
  (never a URL, host, path, filename, embedding or image byte). Fields:
  `endpoint`, `outcome` (`match`/`no_match`/`ok`/`error`), `error_code`,
  `image` (`reference`/`selfie`/`-`), `match`, `cosine_similarity`,
  `threshold`, `det1`/`det2`, `fetch_ms`, `decode_ms`, `queue_ms`,
  `infer_ms`, `total_ms`, `q1_face_px`/`q2_face_px`, `q2_blur`. See
  README "Logging" for the full per-field meaning.
- **`LOG_LEVEL`**: applies inside the container -- the image's `CMD` runs
  `python -m face_recognition_service.main`, which goes through this
  service's own `__main__` block (`uvicorn.run(..., log_level=settings.log_level)`)
  rather than invoking `uvicorn` directly, so `LOG_LEVEL` in the compose
  `environment:` actually changes what's logged.

**What to watch after a deploy**, grepping the summary logger:

- **`NO_FACE_DETECTED` rate** -- a sudden jump usually means a camera/upload
  regression on the calling application's side, not a service bug; compare
  against a recent baseline rather than an absolute number.
- **`SERVICE_BUSY` count** -- non-zero means requests are being rejected
  because `MAX_CONCURRENT_INFERENCE` is saturated; see "Capacity tuning".
- **`match` rate** on the compare endpoints -- a sustained drop after a
  config or image change (threshold, enhancement mode, model pack) is the
  cheapest signal that something regressed recognition, before waiting on a
  user complaint.

## 7. Capacity tuning

- **`MAX_CONCURRENT_INFERENCE`** (default `1`): concurrent `model.analyze`
  calls allowed at once (`concurrency.InferenceGate`). A request that can't
  get a slot within `INFERENCE_QUEUE_TIMEOUT` gets a fast `503
  SERVICE_BUSY` with `Retry-After: BUSY_RETRY_AFTER_SECONDS` instead of
  piling up indefinitely.
- **`INFERENCE_QUEUE_TIMEOUT`** (default `20.0`s): how long a request waits
  for a free inference slot before that 503.
- **`ORT_INTRA_OP_THREADS`** (compose default `2`, matching
  `docker-compose.yml`'s `cpus: 2` limit) and **`ORT_INTER_OP_THREADS`**
  (compose default `1`): ONNX Runtime's own thread pool sizes. Leaving
  these at the code default (`0` = runtime picks from host core count) is
  wrong inside a CPU-limited container -- it oversizes the pool relative to
  what's actually available and can hurt latency under load. Match them to
  whatever CPU limit you set in `deploy.resources.limits.cpus`
  (`FACE_RECOGNITION_CPU_LIMIT`).

**The 60s budget**: queue wait (`INFERENCE_QUEUE_TIMEOUT`, default 20s) +
reference fetch (`FETCH_TOTAL_TIMEOUT`, default 20s, plus DNS/scheduling
slop) + inference itself (well under a minute) has to stay under whatever
client timeout the calling application uses, or it gives up and shows an
outage before this service has had a chance to answer. Don't raise the
queue or fetch timeouts without checking that budget still fits.

**On-VPS benchmark procedure** (not run in this repo -- there is no Docker
on the dev machine):

1. Have a stored reference photo URL reachable from the VPS (or a selfie
   upload) and a selfie image ready.
2. Time N sequential `/compare-photos` calls -- `image1` is the reference
   *URL* as a form field (not a file part), `image2` is the uploaded selfie
   file (see `face_recognition_service/main.py`'s `compare_photos` route):
   ```bash
   for i in $(seq 1 20); do
     time curl -s -o /dev/null -w "%{http_code}\n" \
       -H "Authorization: Bearer $API_TOKEN" \
       -F "image1=https://your-reference-host/photo.jpg" \
       -F "image2=@selfie.jpg" \
       https://your-domain.com/api/v1/compare-photos
   done
   ```
   To benchmark with two local files instead of a reference URL, use
   `/compare-photos-upload` with both fields as file uploads instead:
   `-F "image1=@reference.jpg" -F "image2=@selfie.jpg"`.
3. Run once with `ORT_INTRA_OP_THREADS=1`, once with `=2` (both under the
   same `cpus: 2` compose limit), and compare `total_ms` from the summary
   log across the two runs, not just wall-clock `time`.
4. Use whichever setting gives lower median latency at the concurrency
   level you actually expect; more threads isn't always faster under a
   hard CPU limit.

## 8. Security notes

- **No liveness/anti-spoofing check.** This service verifies that a
  photo's face matches a reference embedding; it does not verify the photo
  was taken of a live person in front of the camera at capture time. A
  still photo of a photo, or of a screen, can pass if it clears the
  detection/quality/match thresholds. If your use case requires proof of
  liveness, that has to be added separately (client-side capture
  constraints, a dedicated liveness model, etc.) -- it is explicitly out of
  scope here.
- **InsightFace's pretrained weights (including antelopev2) are
  non-commercial research licensed.** A commercial deployment needs either
  a commercial license from InsightFace or different, commercially-licensed
  weights -- this is a licensing decision for whoever operates the
  deployment, not something this guide can resolve.
- **Residual SSRF risks** in the reference-URL fetch, even with hosts
  allowlisted and private addresses blocked (see README "Reference URL
  host/IP policy" for the full mechanism):
  - **DNS-rebinding TOCTOU**: the host is resolved and checked, then
    requested moments later; it could resolve to an allowed address at
    check time and a different, blocked one by the time the actual request
    connects. Narrow window, not eliminated.
  - **Resolver time is unbounded by `FETCH_TOTAL_TIMEOUT`**: each hop's
    `getaddrinfo` call (up to two lookups, A and AAAA) has no timeout of
    its own and isn't interruptible by the fetch watchdog, so a slow
    resolver adds on top of the stated timeout rather than counting
    against it.
- **TLS verification when calling out**: whatever calls this service
  should itself verify TLS certificates on any HTTPS reference-photo URLs
  it hands over; this service does not change or weaken that on its own.

## 9. Troubleshooting

Every `ErrorCode` the face-processing/fetch/inference path can raise (plus
the two generic HTTP-layer codes an operator is most likely to hit), its
HTTP status, whether it's tagged with which photo (`image: "reference"` /
`"selfie"` / `null`), and what a caller sees / how to fix it.

| Code | HTTP | `image` tag | What the caller sees | Fix |
|---|---|---|---|---|
| `NO_FACE_DETECTED` | 400 | reference/selfie | No face found in the image at all. | Ask for a clearer retake. |
| `FACE_LOW_QUALITY` | 400 | reference/selfie | The detector's own confidence gate (`MIN_FACE_QUALITY`, always on) failed, or an opt-in quality gate did (all off by default). | Ask for a clearer retake. If it's spiking with gates still off, check `MIN_FACE_QUALITY`/`DETECTION_THRESHOLD` haven't drifted from the recommended values (§2). |
| `MULTIPLE_FACES_DETECTED` | 400 | reference/selfie | Only possible if `MULTI_FACE_POLICY=reject_ambiguous` is set (default `largest` never returns this). | Don't enable `reject_ambiguous` without confirming the calling application recognizes this code -- otherwise it's shown as a generic outage there. |
| `INVALID_IMAGE` | 400 | reference/selfie | Image decode/format error, or a malformed reference URL (bad scheme/no host) on the *original* URL. | Check the upload/URL is actually a valid image. |
| `UNSUPPORTED_FORMAT` | 400 | reference/selfie | Image format not in `ALLOWED_IMAGE_FORMATS`. | Convert to jpg/png/webp/bmp/mpo. |
| `IMAGE_TOO_LARGE` | 400 | reference/selfie | Encoded image exceeds `MAX_IMAGE_SIZE`, or the whole request body exceeds `MAX_REQUEST_BODY_BYTES` (untagged in the body-size-middleware case). | Compress/resize before upload. |
| `REFERENCE_UNAVAILABLE` | 400 | reference | The reference URL itself returned a definite 4xx (404/403/410/...). | The reference photo/link needs replacing at the source. |
| `SERVICE_UNAVAILABLE` | 503 | none (untagged) | The reference URL couldn't be *reached* -- timeout, DNS failure, TLS/connection error, or a 5xx from the far end. A fetch-path fault, not evidence the link is bad. | Check upstream reachability from the VPS, not the reference photo. |
| `SERVICE_BUSY` | 503 | none (untagged) | `MAX_CONCURRENT_INFERENCE` slots all busy and `INFERENCE_QUEUE_TIMEOUT` elapsed. | See "Capacity tuning": raise `MAX_CONCURRENT_INFERENCE`/CPU, or investigate why inference is slow. |
| `REFERENCE_URL_NOT_ALLOWED` | 400 | none (untagged) | Reference URL's host (or a redirect target's) failed the host/IP allowlist or the always-blocked-address check. | Check `REFERENCE_URL_ALLOWED_HOSTS` includes the real host(s) (§3), or that this isn't someone probing internal addresses. |
| `MODEL_NOT_LOADED` | 503 | none | Model initialization failed (bad model files, OOM during load, etc.). | Check `docker compose logs face-recognition` for the load error; verify the baked-in antelopev2 pack wasn't corrupted. |
| `INVALID_EMBEDDING` | 400 | n/a | Embedding validation failed on `/compare` (wrong dimensionality, non-finite values). | Only reachable via the embedding-input endpoints. |
| `PROCESSING_ERROR` | 500 | none | Unexpected server-side fault. | Check `docker compose logs face-recognition` and the request's `X-Request-ID` for the stack trace. |
| `UNAUTHORIZED` | 401 | none | Missing/invalid bearer token -- `API_TOKEN` mismatch between this service and the caller. | Re-sync `API_TOKEN` on both sides after any rotation. |
| `INVALID_REQUEST` / `VALIDATION_ERROR` | 400 / 422 | none | A malformed request the router itself rejects before reaching face-processing code -- missing required multipart field, wrong `distance_metric` value, wrong content type, etc. | Usually a caller-side integration bug (a field renamed/dropped) rather than a runtime fault; check the request actually sent against `face_recognition_service/main.py`'s route signatures. |
| Plain `413` from nginx (not this service) | 413 | n/a | `nginx/nginx.conf`'s `client_max_body_size` (32M) is smaller than the request. | Not a code this service ever emits -- raise `client_max_body_size` together with `MAX_REQUEST_BODY_BYTES` if you ever increase the latter, keeping nginx's cap **strictly above** the app's (they must not be set equal -- see the comment in `nginx/nginx.conf`). If you see this, nginx is rejecting before this service's own `MAX_REQUEST_BODY_BYTES` 400 gets a chance to. |

**Health check returns 503 right after a deploy**: expected until the model
finishes loading (§5); if it stays 503 past the ~60s `start_period`, check
`docker compose logs face-recognition` for a model-load failure
(`MODEL_NOT_LOADED`).

**`docker compose config` / `up` fails with "API_TOKEN must be set"**:
expected -- see §2. Export `API_TOKEN` or put it in `.env`.

**CORS preflight fails from a browser**: expected -- `CORS_ENABLED=false`
by default, since the intended caller is server-to-server. This is not a
bug to fix by enabling CORS unless you actually have a browser-based
caller.

**Certificate never issues / certbot loops with errors**: check
`cloudflare.ini` exists, has correct permissions (`chmod 600`), and its
token has `Zone.Zone:Read` + `Zone.DNS:Edit` on the right zone; check
`DOMAIN_NAME` and the `{{DOMAIN_NAME}}` substitution in `nginx/nginx.conf`
actually match a domain that exists in the Cloudflare account.
