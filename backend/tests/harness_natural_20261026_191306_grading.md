# Natural gold set — grading sheet

- run at: 2026-10-26 19:13:06.398480+00:00
- backend: http://localhost:8000
- question file: `tests\gold_natural.json`, tag: `baseline_7b_local2`
- completed: 7/63

Auto-checks are mechanical (cited file and page, script, fixed refusal string). They do not judge the answer; fill in Correct (Y/P/N) by hand.

## Summary

| check | result | over |
|---|---|---|
| citation_file_hit | 85.7% (6/7) | answerable questions |
| citation_page_hit | 85.7% (6/7) | answerable questions |
| top_citation_is_expected | 85.7% (6/7) | answerable questions |
| language_match | 100.0% (6/6) | all questions, infra failures excluded |
| correct_refusal | n/a (0/0) | negative_trap questions, infra failures excluded |
| false_refusal | 0.0% (0/6) | answerable questions, infra failures excluded |
| median seconds per question | 61.52 | all questions, infra failures excluded |
| max seconds per question | 106.8 | all questions, infra failures excluded |
| excluded (infra_failure) | 1 | all questions |
| errors | 1 | all questions |

## Answers

| id | tier | language | question | expected file and page | cited files and pages | answer (first 400 characters) | Correct (Y/P/N) | Notes |
|---|---|---|---|---|---|---|---|---|
| nat-001 | situation | en | Nobody runs a proper library in our area and the local trust refuses to; can the State Government open a public library there itself? | GR1.1.pdf p7 | GR1.1.pdf p1; GR1.2.pdf p1; GR1.1.pdf p7; GR1.1.pdf p6; GR1.1.pdf p9; GR1.2.pdf p4; GR1.1.pdf p8; GR1.1.pdf p5 | - Yes, the State Government can open a public library in the area itself. - According to the Maharashtra Public Libraries Act, 1967, if the local authority, society, or trust is unwilling or incompetent to provide library service to the public, the State Government may establish a library for the use of the public in that area. Specifically, Section 11 of the Act states that no such library shall |  |  |
| nat-002 | detail | en | Under the public libraries law, what is the minimum sum the State Government must put into the Library Fund every year? | GR1.1.pdf p9 | GR1.1.pdf p1; GR1.1.pdf p9; GR1.1.pdf p7; GR1.1.pdf p3; GR1.1.pdf p5; GR1.1.pdf p6; GR1.1.pdf p10; GR1.1.pdf p8 | - The State Government shall, after due appropriation made by law in this behalf, contribute to the Library Fund every year, a sum not less than twenty-five lakhs of rupees. |  |  |
| nat-003 | detail | en | How many times a year must the State Library Council meet at minimum, and how long may pass between two meetings? | GR1.1.pdf p6 | GR1.1.pdf p1; GR1.2.pdf p1; GR1.1.pdf p6; GR1.1.pdf p5; GR1.2.pdf p4; GR1.1.pdf p8; GR1.2.pdf p3; GR1.2.pdf p2; GR1.1.pdf p4 | - The State Library Council must meet at least twice a year. - Six months shall not intervene between two successive meetings. |  |  |
| nat-004 | situation | en | I only hold a certificate in library science; how much experience do I need to be nominated as librarian on a District Library Committee? | GR1.2.pdf p4 | GR1.2.pdf p1; GR1.1.pdf p1; GR1.4.pdf p1; GR1.2.pdf p4; GR1.1.pdf p8; GR1.2.pdf p5; GR1.1.pdf p10; GR1.1.pdf p6; GR1.1.pdf p9; GR1.1.pdf p11; GR1.2.pdf p2 | - To be nominated as a Librarian on a District Library Committee, you need to have three years of experience as a Librarian, in addition to holding a certificate in Library Science. This is according to the rules specified in clause (ii) of sub-section (1) of section 11 of the Maharashtra Public Libraries Act, 1967. |  |  |
| nat-005 | detail | en | Within how many days must the Secretary send the minutes of a State Library Council meeting to the presiding authority for approval? | GR1.2.pdf p3 | GR1.2.pdf p1; GR1.1.pdf p1; GR1.2.pdf p3; GR1.1.pdf p6; GR1.2.pdf p4; GR1.1.pdf p5; GR1.2.pdf p2; GR1.1.pdf p8 | - The Secretary shall submit the minutes of the meeting within fifteen days to the Presiding authority for approval. |  |  |
| nat-006 | detail | en | How often, at minimum, is a District Library Committee required to meet under the business procedure rules? | GR1.2.pdf p4 | GR1.2.pdf p1; GR1.1.pdf p1; GR1.2.pdf p4; GR1.2.pdf p5; GR1.2.pdf p3; GR1.2.pdf p2; GR1.1.pdf p6; GR1.1.pdf p8 | - Every District Library Committee shall meet at least once in a year on a date to be fixed by the President. |  |  |
| nat-007 | situation | mr | ग्रंथालय परिषदेत नव्याने सहा सदस्य घेतले असे ऐकले; त्यात विधानसभेचे कोणते आमदार समाविष्ट केले आहेत? | GR1.pdf p2 | — | ERROR: ReadTimeout: timed out |  |  |
