# -*- coding: utf-8 -*-
import sys; sys.path.insert(0, '.')
from engine.interpreter import answer_research_question, generate_viva_questions

ctx = {'last_result': {}, 'dataset_profile': {}, 'cleaning_log': []}
a = answer_research_question('why point biserial correlation?', ctx)
print('POINT BISERIAL ANSWER (first 300):')
print(a[:300])
print()

# Check what viva questions are generated for Point-Biserial
analysis_history = [{'test': 'Point-Biserial Correlation', 'r_pb': 0.41, 'p_value': 0.001, 'n': 80}]
qs = generate_viva_questions(analysis_history)
print('ALL Point-Biserial questions:')
for q in qs:
    print('  Q:', q['question'][:80], '| Cat:', q['category'])
print()

# GoF questions
analysis_history2 = [{'test': 'Chi-Square Goodness-of-Fit', 'chi2': 6.3, 'df': 2, 'p_value': 0.043, 'cohens_w': 0.23}]
qs2 = generate_viva_questions(analysis_history2)
print('ALL GoF questions:')
for q in qs2:
    print('  Q:', q['question'][:80], '| Cat:', q['category'])
