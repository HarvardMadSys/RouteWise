"""Online routing strategies for experiment simulation.

This subpackage contains online decision-making strategies that do not have
access to future request information. They can be evaluated in offline
simulation mode by replaying historical traces.

Strategies support both Stage1 (S_Q only) and Stage2 (S_Q + S_C) through
configuration-based strict gating.

Algorithms:
- Greedy: FCFS baseline, uses subscription whenever available
- PrimalDual: Threshold-based with competitive ratio guarantees
- LearningAugmented: ML-enhanced Primal-Dual with quantile prediction
"""

from experiment.strategies.online.base import OnlineStrategy
from experiment.strategies.online.greedy import (
    GreedyCostAwareStrategy,
    GreedyOnlineStrategy,
)
from experiment.strategies.online.learning_augmented import (
    LAPDConfig,
    LearningAugmentedPrimalDualStrategy,
    LearningAugmentedUnifiedStrategy,
)
from experiment.strategies.online.primal_dual import (
    CAPQConcurrencyManager,
    PrimalDualOnlineStrategy,
    PrimalDualQuotaManager,
)

__all__ = [
    # Base
    "OnlineStrategy",
    # Greedy strategies
    "GreedyOnlineStrategy",
    "GreedyCostAwareStrategy",
    # Primal-Dual strategies
    "PrimalDualQuotaManager",
    "CAPQConcurrencyManager",
    "PrimalDualOnlineStrategy",
    # Learning-Augmented strategies
    "LAPDConfig",
    "LearningAugmentedPrimalDualStrategy",
    "LearningAugmentedUnifiedStrategy",
]
