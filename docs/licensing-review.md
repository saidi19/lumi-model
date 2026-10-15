# Source licensing review — holder confirmed, license review pending

No root source-code license was found on the audited default branch. This review
branch adds the official Apache-2.0 text as `LICENSE` and the confirmed holder
in `NOTICE` for review before merging. No merge is performed automatically.

Proposed source license: Apache License 2.0. Copyright holder for the
maintainer-owned original source: **Md. Delower Hossain Saidi**.
The maintainer explicitly confirmed personal ownership and this full name on
2026-10-15. This replaces the earlier company-holder proposal; existing company
branding and package author metadata do not establish copyright ownership.
This confirmation does not claim ownership of third-party contributions or
assets and does not itself activate the proposed license.

Before activation:

- Personal holder confirmation is complete. Verify contribution provenance and
  retain third-party ownership and notices.
- The main source/scripts/tests notice scan found no existing copyright or
  copied-code attribution; absence of a notice is not proof of original authorship.
  The maintainer's ownership confirmation applies to their original contributions.
  The development branch contains KaTeX under its retained MIT notice, attributed
  to Khan Academy and other contributors. It is not merged or relicensed here.
- Define the source files covered by the grant and any exclusions.
- Review benchmark-signature redistribution separately.
- Review the supplied official license text and holder notice before merging.

## Dependency inventory

Installed distribution metadata inspected during this audit reports:

| Dependency/version inspected | License metadata |
| --- | --- |
| torch 2.10.0 | BSD-3-Clause |
| transformers 5.2.0 | Apache-2.0 |
| tokenizers 0.22.2; safetensors 0.7.0 | Apache license classifiers |
| pytest 9.0.2; ruff 0.15.1; datasketch 2.0.0 | MIT |
| lingua-language-detector 2.2.0; pyarrow 25.0.0 | Apache-2.0 |
| zstandard 0.25.0 | BSD-3-Clause |
| huggingface-hub 1.30.0 installed runtime | Apache-2.0; not the optional pinned 1.23.0 release |

These dependencies are not vendored or relicensed by this PR. Package metadata is
not a complete transitive notice audit. Optional MLX/MLX-LM and the pinned
benchmark-build runtime were not installed or cleared for binary redistribution.
Distributors must retain bundled third-party notices and review their exact builds.
No observed source-license conflict is established by this inventory; it is not
a guarantee covering every optional package or asset.

Official license text: https://www.apache.org/licenses/LICENSE-2.0.txt

Model weights, training data, tokenizer assets and private evidence are not
automatically covered by the proposed source license. Each requires its own
rights and distribution decision. The current review must not be advertised as
completed legal clearance or a fully licensed open-source release.
