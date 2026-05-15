import os, sys

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.dirname(here)
print('HERE=', here)
print('REPO=', repo)
print('sys.path[0:5]=', sys.path[:5])

print('MODEL candidates:')
for p in [
    os.path.join(repo,'models','global_model.pth'),
    os.path.join(repo,'models','global_model.pkl'),
    os.path.join(repo,'global_model.pth'),
    os.path.join(repo,'global_model.pkl'),
]:
    print(' -', p, os.path.exists(p))

print('PLOTS candidates:')
for p in [
    os.path.join(repo,'outputs','analysis_outputs','federated_accuracy_plot.png'),
    os.path.join(repo,'outputs','analysis_outputs','federated_metrics_plot.png'),
    os.path.join(repo,'outputs','analysis_outputs','training_results.png'),
    os.path.join(repo,'analysis_outputs','federated_accuracy_plot.png'),
    os.path.join(repo,'analysis_outputs','federated_metrics_plot.png'),
    os.path.join(repo,'analysis_outputs','training_results.png'),
]:
    print(' -', p, os.path.exists(p))

