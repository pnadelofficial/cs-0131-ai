#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# version 2.0.1 - copyright (c) 2023-2024 Santini Fabrizio. All rights reserved.
#

import bt_library as btl
import bt as bt
from turtle import Turtle

# Instantiate the tree according to the assignment. The following are just examples.

tree_root = bt.Priority(
    [
        bt.Sequence(
            [
                bt.BatteryLessThan30(),
                bt.FindHome(),
                bt.GoHome(),
                bt.Dock()
            ],
        ),
        bt.Selection(
            [
                bt.Sequence(
                    [
                        bt.SpotCleaning(),
                        btl.Timer(20, bt.CleanSpot()), 
                        bt.DoneSpot()
                    ]
                ),
                bt.Sequence(
                    [
                        bt.GeneralCleaning(),
                        bt.Sequence(
                            [
                                bt.Priority(
                                    [
                                        bt.Sequence(
                                            [
                                                bt.DustySpot(),
                                                btl.Timer(35, bt.CleanSpot()),
                                                bt.AlwaysFail()
                                            ]
                                        ),
                                        bt.UntilFail(bt.CleanFloor())
                                    ], 
                                    [1, 2]
                                ),
                                bt.DoneGeneral()
                            ]
                        )
                    ]
                )
            ]
        ),
        bt.DoNothing()
    ],
    [1, 2, 3]
)
