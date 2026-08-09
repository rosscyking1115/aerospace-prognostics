# Departures from the house documentation standard

This repository is documented to the shared house style used across these public
repositories. In three places that standard is wrong *for this project*, and this repo
departs from it deliberately. Each departure is recorded here rather than applied
silently, because an undocumented departure is indistinguishable from an oversight.

Each item below is written so it can be lifted into the standard verbatim. Three further
notes at the end record findings whose class is wider than this repository: two defects and
one pattern worth adopting.

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

## Three notes whose class is wider than this repository

Recorded here for the same routing. None is a departure from the standard; all three are
gaps the standard does not currently cover. The first two are defects; the third is a
pattern worth adopting.

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

**A note about a temporary state should name its own deletion condition.** Recording a
known gap is the right move when closing it costs more than it buys — but a note that
describes a temporary state and does not say when it stops being true becomes permanent by
default. Nobody deletes a comment they are not sure is finished, so it accretes, and a file
of stale notes teaches readers that the notes are not maintained, which costs more than the
one note was ever worth.

The fix is one sentence: state the condition under which the note should be removed, and
put it where the person who satisfies that condition will be standing. Worked example here,
`.github/workflows/ci.yml`: the advisory-not-blocking path had never run, and rather than
manufacture a vulnerable pin to exercise it, the gap was recorded with the instruction to
verify the behaviour when a real advisory lands — *before* fixing the advisory, because
fixing it first destroys the only natural test case the path will get — and then to delete
the note. It appears in three places ordered by when someone meets them: the job summary the
failed run prints, the workflow comment beside the reasoning, and the contributing guide's
advisory procedure. Three copies because the person who hits it will be trying to fix
something else, and an instruction filed where it is not being read is not an instruction.

**Proposed wording for the standard:**

> A note recording a temporary state — an untested path, a pending upstream fix, a
> deliberate gap — must state the condition under which it should be deleted. Without one it
> is permanent by default, because nobody removes a note they cannot tell is finished. Put
> the note where the person who satisfies the condition will be standing, not only where it
> is tidy to file it.
