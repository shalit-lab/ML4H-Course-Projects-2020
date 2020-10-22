from typing import List

from matplotlib import pylab as plt
from matplotlib.patches import Polygon
from result import FitResult

def merge_results(results: List[FitResult]):
    train_losses = [result.train_loss for result in results]
    train_accs = [result.train_acc for result in results]
    test_losses = [result.test_loss for result in results]
    test_accs = [result.test_acc for result in results]
    merged_result = FitResult(
        num_epochs=results[0].num_epochs,
        train_loss=list(zip(*train_losses)),
        train_acc=list(zip(*train_accs)),
        test_loss=list(zip(*test_losses)),
        test_acc=list(zip(*test_accs))
    )
    return merged_result