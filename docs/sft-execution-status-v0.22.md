# SFT execution status v0.22

Local engineering checkpoint, 2026-09-07. A fixed-host HTTPS transport library
and bounded HTTP framing profile are implemented. All exercised transport uses
fake DNS, sockets, TLS contexts and credentials. No provider request, account
lookup, instance action or GPU execution occurred. Gate 5 remains partial.

## Previous step verified

Durable-journal commit `c5bb3cfa7a023742e98edd7b0a66c77b6be4f58a` was pushed.
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34081626159) passed
all four jobs: 3,271 passed, 58 existing skips and 329 separately routed/deselected
nodes, with 6,609 subtests. Source selection accounted for 3,329 nodes; 327 private
nodes remained NOT EXECUTED.
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34081626153) passed
all three synthetic tests with zero skips. Both completed run conclusions and
their exact commit were rechecked. These are not full private integration results.

## Scope and authority boundary

Only five new versioned files: HTTP request/framing module, HTTPS transport,
their two synthetic test modules and this checkpoint. Historical sources, tests,
plans, evidence, launch contract, local budget, workflows and inventories are
unchanged. No API credential is read from disk, environment, browser or account.
Only a caller-supplied transient token is accepted by the pure request compiler.

IMPORTANT: `LambdaHttps.exchange()` is real-network-capable code, not an inert
decision model. It MUST NOT be exercised with real dependencies under the frozen
`local_decision_model_only` plan or the current incomplete launch contract.
There is no CLI, live test, automatic invocation or durable-controller bridge.
A separately authorized external controller is still required. Neither a request
object nor a supplied digest authenticates owner approval, account identity,
target ownership, an execution ticket or global exclusivity.

The public [Lambda API specification](https://docs-api.lambda.ai/api/cloud/spec.json)
was checked for the primary host, Bearer authentication, launch and targeted
terminate operations. The observed 249,558-byte document had SHA-256
`2e00f2884d043fa2377a1a6f898eba4b81d8b0c4546d5d98079c7faa4451ba8f`.
Public documentation access is distinct from authenticated account traffic.
An additional schema refresh was denied by the approval service's credit error;
it was not retried or bypassed. No live provider compatibility is claimed.

## Fixed request profile

The compiler accepts only three operations on `cloud.lambda.ai:443`:

| Operation | Method and path | Body restriction |
| --- | --- | --- |
| Create | POST `/api/v1/instance-operations/launch` | Canonical, SHA-256-pinned launch object |
| Observe | GET `/api/v1/instances/<exact-id>` | Empty |
| Terminate | POST `/api/v1/instance-operations/terminate` | One exact ID in `instance_ids` |

The local launch profile requires exactly one printable ASCII SSH key name,
a canonical lowercase hyphenated image UUID, and exact pinned region/type.
These are intentionally narrow local restrictions, not a claim that the API
accepts no other forms. No implicit image, quantity, mounts, tags, cloud-init,
firewall changes or extra fields are accepted. JSON must be canonical ASCII,
duplicate-free and at most 4,096 bytes. Instance targets are 32 lowercase hex
characters. The actual account's image/key compatibility remains untested.

Headers are fixed, including Bearer authentication, JSON content type, identity
encoding, exact Content-Length and Connection: close. There is no caller-supplied
host, path, proxy or header. CR/LF and invalid/oversized token values reject.
Compiled request size is at most 8,192 bytes. A request is consumed before I/O;
reuse or byte mutation rejects, and its stored raw reference is cleared even on
claim failure. This cooperative object is not a security boundary against
malicious same-process code, and clearing references is not secure memory erasure.

## HTTPS and response bounds

Each exchange calls the IPv4 resolver once for the fixed host and selects only
the first public address. There is one socket creation, TCP connect, explicit TLS
handshake and application send call. No address fallback, proxy, redirect,
reconnect, resend or application-level retry is implemented. A short send is an
ambiguous failed exchange, never permission to submit again.

TLS uses a verified client context, hostname checking, TLS 1.2 minimum, strict
certificate checks and the trusted runtime's default CA roots. CA configuration
is part of that trusted runtime, not independently authenticated by this library.
The context is constructed directly with key logging disabled: Python's
convenience factory can honor `SSLKEYLOGFILE` before returning a context.
The [Python SSL documentation](https://docs.python.org/3/library/ssl.html)
describes certificate verification, client contexts and key-log behavior.
SNI is fixed to the provider host, ALPN is HTTP/1.1, and negotiated TLS/peer
certificate checks precede sending. Ragged TLS EOF is not suppressed.

Only one HTTP/1.1 200 JSON-content-type message is accepted, followed by clean
stream EOF. Content-Length or bounded chunked framing is mandatory. Duplicate
headers, ambiguous framing, compression, interim/error/redirect responses,
folded headers, chunk extensions/trailers, truncation and trailing data reject.
This intentionally strict subset is not a general
[HTTP/1.1 implementation](https://www.rfc-editor.org/rfc/rfc9112.html).
The incremental parser bounds headers to 8,192 bytes, decoded body to 65,536,
wire plaintext to 81,920 and nonzero chunks to 128. Fragmentation is supported.

An exchange prepays 81,921 application read-capacity bytes, including one overflow
probe byte, and 8,192 application write-capacity bytes before DNS. At most 64
exchanges reserve 5,242,944 read and 524,288 write bytes cumulatively. Failed
reservations are not refunded. Each receive requests at most 4,096 bytes and
never exceeds the unused portion of that exchange's prepaid capacity. Short
fragments consume only returned bytes within the existing reservation. At most
256 receive calls include the EOF probe; excessive fragmentation fails closed.

Monotonic deadlines are checked around blocking stages, with a maximum 15-second
exchange window and refreshed per-operation timeouts. Clock regression, wrong
process/thread, reentry, timeout, parser error or cleanup failure permanently
poisons the transport. An owned socket gets one close attempt, with no retry.
Success is withheld until cleanup and a post-close deadline check succeed.
Parser cleanup failure cannot skip the owned socket's close attempt.

These are application-level bounds. Resolver internals, TCP retransmission, TLS
records, system CA access and blocking OS/close calls are not physical byte or
hard wall-time enforcement. A timed-out create may already have reached the
provider. A fresh transport object is not authority to retry it. Actual controller
survival, reconciliation, targeted teardown and billing stop remain unproved.

Successful framing returns transient body bytes only. It does not validate JSON
semantics or prove instance creation, ownership or termination. The frozen
`posttraining_sft_lambda_wire_v1` parser and durable lifecycle validation remain
required at the future integration boundary. Response bodies, IDs, credentials
and exception text are absent from the fixed counter summary; repr is redacted.
No raw network logging or result persistence is provided.

## Verification checkpoint

| Check | Result |
| --- | --- |
| Pure HTTP request and framing tests | 17 passed |
| Fake-only HTTPS transport and fault tests | 39 passed |
| Both new modules under ordinary pytest | 56 passed; 146 subtests passed |
| New plus previous lifecycle/observer/guard/ledger/credential/status/IPC suites | 633 passed |
| Frozen safety/control suites | 273 passed; no skips |
| Exact Ruff on all four new Python files | Zero findings |
| Frozen lint | 815 historical findings/163 files unchanged; zero new |
| Private readiness | 327 OPEN; zero private tests executed |

There are 906 unique targeted tests including frozen controls. Repeated test
environments and subtests do not add unique test cases. DNS, socket creation and
TLS context creation are replaced before every exercised exchange; CA loading
is a mock method. No fake handshake is represented as live TLS evidence.
Tests include 1-byte fragmentation, maximum body/call/capacity limits, DNS and
certificate failures, one-shot sends, redirects, ragged EOF, every blocking-stage
deadline, close faults, reentry and fixed redacted counters.

This is a pre-publication local checkpoint. The new commit's Source and Spark CI
must be checked separately; their outcome is not preclaimed here.

## Next boundary

Next is explicit external-controller integration: fresh authority/account/target
bindings, transport dispatch only after durable reservation, semantic response
validation, failure/reconciliation handling and independent shutdown control.
No live connection is approved merely by this source checkpoint. Current account
image/key/price data, complete execution authority and protected-instance bindings
are still missing. Do not provision a GPU or touch the existing busy instance.

Real host/boot/subtree ownership, aggregate resource enforcement, the real-worker
profile, authentic S3/DCP import, private-data relocation, CUDA acceptance and
the complete execution ticket remain. The S3-backed pretrained Lumi base is
unchanged. Roadmap: gates 1-4 complete, Gate 5 partial, gates 6-8 pending. The 327
private OPEN nodes are not roadmap steps. Production eligibility remains false.

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
