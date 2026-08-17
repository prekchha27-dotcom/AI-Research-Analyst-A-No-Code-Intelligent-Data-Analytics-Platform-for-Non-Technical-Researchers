# -*- coding: utf-8 -*-
"""
Test: Viva prep questions for all new test types + METHOD_EXPLANATIONS + chatbot handlers.
"""
import sys
sys.path.insert(0, '.')

from engine.interpreter import answer_research_question, generate_viva_questions

PASS = 0
FAIL = 0

def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        print(f'  PASS  {name}')
        PASS += 1
    else:
        print(f'  FAIL  {name}' + (f' — {detail}' if detail else ''))
        FAIL += 1

ctx = {'last_result': {}, 'dataset_profile': {}, 'cleaning_log': []}

print('\n--- METHOD_EXPLANATIONS: new test types via chatbot ---')

a = answer_research_question('why did you use welch anova?', ctx)
check('why welch anova — returns content', len(a) > 50)
check('why welch anova — mentions Welch', 'welch' in a.lower())

a = answer_research_question("why fisher exact test?", ctx)
check('why fisher exact — returns content', len(a) > 50)
check('why fisher exact — mentions Fisher', 'fisher' in a.lower())

a = answer_research_question("why point biserial correlation?", ctx)
check('why point biserial — returns content', len(a) > 50)
check('why point biserial — mentions biserial', 'biserial' in a.lower() or 'point' in a.lower())

a = answer_research_question("why goodness of fit test?", ctx)
check('why goodness of fit — returns content', len(a) > 50)
check('why goodness of fit — mentions goodness', 'goodness' in a.lower())

a = answer_research_question("why did you use friedman test?", ctx)
check('why friedman — returns content', len(a) > 50)
check('why friedman — mentions Friedman', 'friedman' in a.lower())

a = answer_research_question("why pca?", ctx)
check('why pca — returns content', len(a) > 50)
check('why pca — mentions PCA or principal', 'pca' in a.lower() or 'principal' in a.lower())

print('\n--- generate_viva_questions: all new test types ---')

analysis_history = [
    {"test": "Welch's One-Way ANOVA", "f_statistic": 4.2, "p_value": 0.02,
     "eta_squared": 0.09, "n_groups": 3, "group_summary": {}},
    {"test": "Fisher's Exact Test", "p_value": 0.03, "odds_ratio": 2.5,
     "phi": 0.22, "ci_95": (1.1, 5.8), "n": 80},
    {"test": "Friedman Test", "statistic": 8.4, "p_value": 0.015,
     "kendalls_w": 0.28, "n_conditions": 3, "n": 80},
    {"test": "Principal Component Analysis (PCA)", "kmo_approx": 0.68,
     "kaiser_n_components": 2, "eigenvalues": [2.1, 1.3, 0.6],
     "explained_variance_ratio": [0.525, 0.325, 0.15], "n": 100},
    {"test": "Point-Biserial Correlation", "r_pb": 0.41, "p_value": 0.001, "n": 80},
    {"test": "Chi-Square Goodness-of-Fit", "chi2": 6.3, "df": 2,
     "p_value": 0.043, "cohens_w": 0.23, "n": 120},
]

qs = generate_viva_questions(analysis_history)
print(f'  Total viva questions generated: {len(qs)}')

# Welch ANOVA
welch_qs = [q for q in qs if 'welch' in q['question'].lower()
            or 'games-howell' in q['hint'].lower()
            or ('games' in q['hint'].lower() and 'howell' in q['hint'].lower())]
check('Welch ANOVA viva questions (>=3)', len(welch_qs) >= 3,
      f'found {len(welch_qs)}: {[q["question"][:50] for q in welch_qs]}')

# Fisher
fisher_qs = [q for q in qs if 'fisher' in q['question'].lower()
             or 'odds ratio' in q['hint'].lower()
             or 'odds ratio' in q['question'].lower()]
check("Fisher's Exact viva questions (>=3)", len(fisher_qs) >= 3,
      f'found {len(fisher_qs)}: {[q["question"][:50] for q in fisher_qs]}')

# Friedman
friedman_qs = [q for q in qs if 'friedman' in q['question'].lower()
               or "kendall" in q['hint'].lower()]
check('Friedman viva questions (>=3)', len(friedman_qs) >= 3,
      f'found {len(friedman_qs)}: {[q["question"][:50] for q in friedman_qs]}')

# PCA
pca_qs = [q for q in qs if 'pca' in q['question'].lower()
          or 'component' in q['question'].lower()
          or 'kmo' in q['hint'].lower()
          or 'principal' in q['question'].lower()]
check('PCA viva questions (>=3)', len(pca_qs) >= 3,
      f'found {len(pca_qs)}: {[q["question"][:50] for q in pca_qs]}')

# Point-Biserial
pb_qs = [q for q in qs if 'biserial' in q['question'].lower()
         or 'biserial' in q['hint'].lower()
         or 'r_pb' in q['hint'].lower()]
check('Point-Biserial viva questions (>=2)', len(pb_qs) >= 2,
      f'found {len(pb_qs)}: {[q["question"][:50] for q in pb_qs]}')

# Goodness-of-Fit
gof_qs = [q for q in qs if 'goodness' in q['question'].lower()
          or "cohen's w" in q['hint'].lower()
          or 'cohen' in q['hint'].lower() and 'w' in q['hint']]
check("GoF viva questions (>=2)", len(gof_qs) >= 2,
      f'found {len(gof_qs)}: {[q["question"][:50] for q in gof_qs]}')

# General questions present
general_qs = [q for q in qs if q['category'] in ('Limitations', 'Data Cleaning', 'Critical Understanding')]
check('General viva questions (>=4)', len(general_qs) >= 4,
      f'found {len(general_qs)}')

# All questions have required keys
missing_keys = [i for i, q in enumerate(qs)
                if 'question' not in q or 'hint' not in q or 'category' not in q]
check('All questions have question/hint/category', len(missing_keys) == 0,
      f'Missing keys in items: {missing_keys}')

print('\n--- Viva chatbot integration ---')
ctx2 = {
    'last_result': {'test': "Welch's One-Way ANOVA", 'f_statistic': 4.2,
                    'p_value': 0.02, 'eta_squared': 0.09},
    'dataset_profile': {'n_rows': 120},
    'cleaning_log': [],
}
a = answer_research_question("how do I defend welch anova in my viva?", ctx2)
check('Viva defend Welch ANOVA — responds', len(a) > 50)
check("Viva defend Welch ANOVA — mentions test name", "welch" in a.lower() or "anova" in a.lower())

a = answer_research_question("how do I justify fisher exact in my viva?", ctx2)
check('Viva justify Fisher Exact — responds', len(a) > 50)

print()
print('='*60)
print(f'  PASSED : {PASS}')
print(f'  FAILED : {FAIL}')
print('='*60)
if FAIL == 0:
    print('  ALL VIVA TESTS PASSED')
else:
    print(f'  {FAIL} TEST(S) FAILED')
print('='*60)
