# Departures from the house documentation standard

This repository is documented to the shared house style used across these public
repositories. In three places that standard is wrong *for this project*, and this repo
departs from it deliberately. Each departure is recorded here rather than applied
silently, because an undocumented departure is indistinguishable from an oversight.

Each item below is written so it can be lifted into the standard verbatim. Two further
notes at the end record defects found here whose class is wider than this repository.

---

## 1. The PyPI badge is conditional on publishing; the order is what must hold

**The standard says:** badges are "PyPI version · Python versions · Licence · CI status",
same set and same order in every repository.

**Why that is wrong as written:** it makes the PyPI badge unconditional. A PyPI version
badge on a package that has never been published renders as a broken "not found" — the
first thing a reader sees is a badge announcing that the project does not exist on the
index. That is worse than no badge, and it is the standard's own reasoning about number
badges applied one step further.

**Replacement wording:**

> Badges: **PyPI version · Python versions · Licence · CI status**, then a DOI badge if one
> exists. Nothing else. The *order* is fixed and must hold in every repository. Individual
> badges are conditional on the thing they report existing: omit the PyPI badge on a
> package that is not published, and the DOI badge where there is no DOI. An omitted badge
> costs nothing; a badge reporting "not found" is worse than no badge at all.

**Applied here:** this package is not on PyPI and nothing publishes it, so the badge row is
Python · Licence · CI, in the standard's relative order, with PyPI omitted.

---

## 2. The absolute-link rule is scoped correctly but reads as universal

**The standard says:** use absolute URLs for "every link in a README that doubles as a
package description", because a package index resolves no relative targets.

**Why that needs rewording:** the scope is correct and the reasoning is sound, but the rule
sits under a heading that reads as a blanket instruction, and the sentence "there is no
cost to making them all absolute" is not true for a repository whose README never ships as
a long description. Here it would mean rewriting 36 links, and then maintaining them —
absolute links break silently on a rename or a repository move, where relative ones do not.
The rule buys real protection where it applies and pure maintenance where it does not.

**Replacement wording:**

> **This rule applies when, and only when, the README ships as a package's long
> description** — a PyPI or other index page, where no relative target resolves. In that
> case every link must be absolute, including the licence badge, because the index serves
> the rendered description and not a browsable copy of the source tree.
>
> Where a README does not double as a package description, relative links are correct and
> absolute ones are a maintenance cost with no reader benefit: they break on a rename or a
> move where relative links survive. If a repository may publish later, make the links
> absolute at the point publishing is set up, not before — and check the built artifact,
> because this is one of the defects a source-tree check cannot see.

**Applied here:** relative links throughout, recorded rather than defaulted into. If this
package is ever published, the links must be converted first.

---

## 3. The corrections rule names a file the standard never requires

**The standard says:** if a published claim was wrong, say so in the README and record it
in `CHANGELOG.md` as a correction rather than a tidy-up.

**Why that is wrong as written:** nothing in the standard requires a `CHANGELOG.md` to
exist, so in any repository without one the requirement is unenforceable — and a rule that
cannot be followed teaches readers that the rules are decorative. A changelog is also the
wrong instrument for a research repository, where what needs correcting is usually a
*number* and its scope, not a release. A changelog records what changed between versions; a
claims ledger records what each number is allowed to support, which is the thing a
correction actually moves.

**Replacement wording:**

> **Correct in place; never quietly rewrite.** If a published claim was wrong, say so on
> the surface that carried it, and record the correction in the repository's standing
> record of published claims. That record is `CHANGELOG.md` where one exists. Where it does
> not, a claims ledger (`claims.md`) is the acceptable — and for a research repository the
> better — alternative, because a correction usually moves a number's scope rather than a
> release. What is not acceptable is a correction with no durable home.
>
> A visible retraction reads as trustworthy. A silent edit reads as nothing at all, until
> somebody finds the diff — and then it reads as concealment.

**Applied here:** corrections are recorded in [claims.md](../claims.md) and on the page that
carried the number. There is no `CHANGELOG.md`.

---

## Two defects whose class is wider than this repository

Recorded here for the same routing. Neither is a departure from the standard; both are
gaps the standard does not currently cover.

**The artifact is not the repository.** A built sdist here contained local-only files that
no source-tree check could see: they are ignored via a *global* gitignore, which the build
backend does not read, so `git status` is clean and `git ls-files` returns nothing while the
artifact carries them. This is the fifth instance of the same class across these
repositories. The control that works is an allowlist rather than an exclude list — an
exclude list stops only what someone thought to name and fails open the next time a
local-only directory appears, while an allowlist fails closed — plus a test asserting every
allowlist entry matches a tracked path. The verification that works is building the
artifact and opening it. Four earlier instances passed a source check and failed in the
package.

**The twin-surface defect has a second direction: under-claiming.** Where one library is
described by two repositories, the two descriptions drift. Every previous instance across
these repositories was *over*-claiming — a repository asserting more than its dependency
supported. This one ran the other way: this repository recorded a capability as blocked
upstream after the upstream library had already shipped it, so the published limitation was
stricter than the truth. Under-claiming is harder to notice, because nothing about it looks
like a risk, and a reader who believes it simply thinks less of the work than it deserves.
The check is the same in both directions: read what the other repository says about itself,
and reconcile — but it has to be run for understatement as well as overstatement.
