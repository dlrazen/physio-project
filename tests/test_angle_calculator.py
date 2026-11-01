
from src.angle_calculator import AngleCalculator


calculator = AngleCalculator()

test_cases = [
    {
        "name": "Fully extended",
        "shoulder": (0, 0),
        "elbow": (1, 0),
        "wrist": (2, 0),
        "expected": 0
    },
    {
        "name": "90 degree flexion",
        "shoulder": (0, 0),
        "elbow": (1, 0),
        "wrist": (1, 1),
        "expected": 90
    },
    {
        "name": "60 degree flexion",
        "shoulder": (0, 0),
        "elbow": (1, 0),
        "wrist": (1.5, 0.866),
        "expected": 60
    },
    {
        "name": "120 degree flexion",
        "shoulder": (0, 0),
        "elbow": (1, 0),
        "wrist": (0.5, 0.866),
        "expected": 120
    }
]

for case in test_cases:

    result = calculator.calculate_angle(
        case["shoulder"],
        case["elbow"],
        case["wrist"]
    )

    passed = abs(result - case["expected"]) < 0.1

    print(
        f'{case["name"]}: '
        f'Expected={case["expected"]}, '
        f'Actual={result}, '
        f'{"PASS" if passed else "FAIL"}'
    )
