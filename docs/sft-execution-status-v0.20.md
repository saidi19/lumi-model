# SFT execution status v0.20

Local engineering checkpoint, 2026-09-06. The next missing control is a bounded
whole-host lifecycle and targeted teardown workflow. This successor implements
its transport-free decision core and bounded Lambda response parsers. It does
not implement HTTP dispatch, create a host, terminate any instance or authorize
SFT. No GPU is needed for this source-engineering step. Gate 5 remains partial.

## Previous step verified

Read-only observer commit `b98ac55bdceeed9d004547dd937a6056d51f790e` was pushed.
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34041006617) passed
all four jobs: 3,129 tests passed, 58 existing skips, 329 separately routed or
deselected nodes and 5,957 subtests. Source selection accounted for 3,187 nodes;
327 private nodes remained NOT EXECUTED. The mandatory Linux self-observation
executed, but that CI host is not the target SFT host.
[Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34041006650) passed
all three synthetic tests with zero skips. These results do not establish full
private integration, real-worker acceptance or GPU readiness.

## Bounded scope

Only five new versioned files: lifecycle model, Lambda wire projection, their
two synthetic test modules and this checkpoint. Previous source, tests, reports,
workflows, inventories and evidence remain unchanged. The launch contract and
existing local next-run budget remain unchanged and unbound. The existing busy instance
is protected and untouched. There is no inferred instance ID or owner signature.

The existing local next-run budget describes a selected $150 ceiling and 14,400-second
whole-host preparation envelope. Its observed quote is historical, not a current
price check. The model accepts an exact externally pinned plan only in
`local_decision_model_only` mode. It checks declared approval/account/launch-body
hashes, a bounded nonempty protected-instance list, region/type/GPU count, quote,
cost/reserve constraints and a half-open creation time window. These are metadata
checks, not authentication of an approver, account or fresh launch authority.
No populated real plan, credential, execution ticket or owner receipt is created.
The local budget file is already untracked; it is preserved but is not included in
this five-file publication. The new modules do not read or depend on that file.

## Provider semantics checked before implementation

The public [Lambda API browser](https://docs-api.lambda.ai/api/cloud) supplied
its [OpenAPI document](https://docs-api.lambda.ai/api/cloud/spec.json): 249,558
bytes, SHA-256 `2e00f2884d043fa2377a1a6f898eba4b81d8b0c4546d5d98079c7faa4451ba8f`.
Only unauthenticated public documentation was retrieved; no account API was read.
The specification was inspected, not copied into the repository or used as code.

The API exposes a launch response containing an instance-ID array, a direct
instance-detail response and a termination response containing instance objects.
Crucially, the successful termination example has status `terminating`.
Acknowledgment therefore cannot substitute for a later direct observation of
`terminated`. Unknown statuses, 404, empty/multiple results and `preempted` do not
establish completed termination in this local profile.

Lambda documents [one-minute billing increments](https://docs.lambda.ai/public-cloud/billing/).
Billing starts after launch health checks and ends with instance termination;
idle time still counts, and taxes or persistent storage can be separate charges.
The model conservatively projects compute from create dispatch, rounds upward
to whole minutes and integer micro-USD, and includes a declared noncompute reserve.
This is not an enforced provider invoice ceiling or proof of the final invoice.
[Guest shutdown is not provider termination](https://docs.lambda.ai/public-cloud/on-demand/creating-managing-instances/).
The previous process watchdog remains separate from cloud lifecycle control.

## Response boundary

Wire functions have no I/O. One response is at most 65,536 bytes of strict UTF-8
JSON, with no BOM or implicit UTF-16/32 decoding. A pre-parse scanner limits
nesting and containers; duplicate keys at any depth, invalid Unicode, nonfinite
numbers, excessive structures and malformed data reject with one fixed code.
Bounds are depth 16, 128 containers and 1,024 tree items.

Launch requires exactly one 32-character lowercase-hex ID. That is a conservative
local profile, not a claim that Lambda's general string schema guarantees this
format. Details return only ID, known status, region, type, quoted hourly cents
and GPU count. Extra bounded API fields may contain credentials: they are
validated and discarded without logging, not returned or copied into a report.
Transient JSON parsing is not secure erasure. A termination response must name
exactly the requested object and retain its original terminating/terminated state.

## Lifecycle model

All clock samples and responses are explicitly supplied by a trusted caller.
No clock, file, network, process, GPU, private data or credential access occurs.
An immutable intent is an internal decision token, not an HTTP request or approval.

- Create is reserved once before its outcome is known. An ambiguous, malformed,
  late, failed or protected-instance response requires reconciliation; no retry,
  search by name or ownership guess is available.
- A successful single-ID launch binds the only possible teardown target. Work
  remains closed until direct matching active metadata agrees with the pinned
  region, type, GPU count and quote. This modeled work window is not SFT permission.
- Monotonic lifecycle accounting starts at create dispatch and includes boot,
  install, restore, work, export and teardown. The shutdown reserve closes work
  before the hard lifecycle ceiling. Backward clocks and hard-limit exhaustion
  require reconciliation, not a reset or extended budget.
- Request capacity is prepaid cumulatively, with no refunds on failures. Plans
  allow 4-64 requests, each reserving a full 65,536-byte response. Ordinary
  observations cannot consume the last two slots needed for one targeted
  termination and one terminal-status observation.
- Requests have a 15-second modeled deadline, clipped to the whole-host deadline;
  repeated observation dispatches are spaced at least 15 seconds apart. At most
  4,096 state-machine events are admitted. No sleeping, polling loop or scheduler
  is installed by this code.
- A failed or unhealthy bound run still retains its exact-target cleanup
  obligation. Termination is modeled at most once. An ambiguous termination
  outcome can be followed by bounded direct observations, never a second mutation.
- Even a termination response saying `terminated` remains acknowledgment only.
  A separate direct GET of the create-bound object must supply `terminated` for
  modeled terminal evidence. That evidence still does not authenticate transport
  or establish actual termination/billing. Declared creation-window expiration
  does not itself disable cleanup within the original lifecycle budget.

Only aggregate counters, allowlisted states/reasons and explicit modeled booleans
are reported. No instance IDs, raw responses, names, credentials, approval pins
or exception text appear. Actual provider calls/network/GPU/private reads/writes
remain zero, and all real-world authorization/termination/billing flags stay false.

## Verification checkpoint

| Check | Result |
| --- | --- |
| Bounded Lambda response projections | 29 synthetic tests passed |
| Lifecycle, budgets, ordering and failure recovery | 40 synthetic tests passed |
| Both new modules under ordinary pytest | 69 passed; 613 subtests passed |
| New plus prior observer/guard/ledger/credential/lifecycle/status/IPC suites | 504 passed |
| Frozen safety/control suites | 273 passed; no skips |
| Exact Ruff on all four new Python files | Zero findings |
| Frozen lint | 815 historical findings/163 files unchanged; zero new |
| Private readiness | 327 OPEN; zero private tests executed |

There are 777 unique targeted tests including frozen controls; repeated test
environments do not add unique cases. Review-driven regressions verify that
rejection clears pending responses, terminal modeled time/cost freezes, and
advancing into the shutdown reserve cannot revive a reconciliation-only state.
The no-I/O lifecycle test replaces file, descriptor, network, process and clock
entrypoints with failing stubs. All responses and clocks are synthetic.

This is the local pre-publication checkpoint, not a preclaim of the new commit's
Linux Source or Spark result. Those exact CI results must be verified separately.

## Remaining integration boundary

This is a single-threaded volatile decision core, not a durable live controller.
Constructing a second object does not enforce globally exclusive attempts.
A future external controller must authenticate current authority/account/quote,
pin the launch request, use bounded HTTPS without redirects/retries, journal
pre-dispatch reservations durably, survive guest failure, and handle ambiguous
creates through authoritative reconciliation. No missing instance may be selected
from a reusable label. A late or lost result is unresolved, not free compute.
Blocking I/O, provider delays, controller failure and final charges cannot be
bounded by these pure calculations alone. No real execution adapter is enabled.

Keep provisioning off until that live lifecycle workflow and the bounded restore
workflow are ready. Dedicated host/boot/subtree ownership, real aggregate
memory/descendant/network enforcement, the real-worker profile, authentic S3/DCP
import, private-data relocation, CUDA acceptance and the complete execution ticket
remain required. The S3-backed pretrained Lumi base is unchanged.

Roadmap: gates 1-4 complete, Gate 5 partial, gates 6-8 pending. The 327 private OPEN
nodes are not 327 roadmap steps. No SFT, GPU, restore, private input, registry,
promotion, seal, remote login or instance action occurred.

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
