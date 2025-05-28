import evaluate


def main():
    predictions = [
        """Pluto is a dwarf planet in our solar system, located in the Kuiper Belt beyond Neptune, and was formerly considered the ninth planet until its reclassification in 2006."""
    ]
    references = [
        """Pluto is a dwarf planet in the solar system, located in the Kuiper Belt beyond Neptune, and was previously deemed as a planet until it was reclassified in 2006."""
    ]

    rouge = evaluate.load("rouge")
    scenario1_results = rouge.compute(predictions=predictions,
                                      references=references)
    print("Scenario 1:")
    print(scenario1_results)

    generated = [
        "The burrow stretched forward like a narrow corridor for a while, then plunged abruptly downward, so quickly that Alice had no chance to stop herself before she was tumbling into an extremely deep shaft."
    ]
    reference = [
        "The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."
    ]

    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    scenario2_results_bleu = bleu.compute(predictions=generated,
                                          references=reference)
    scenario2_results_meteor = meteor.compute(predictions=generated,
                                              references=reference)
    print("Scenario 2:")
    print(f"Bleu: {scenario2_results_bleu['bleu']}")
    print(f"Meteor: {scenario2_results_meteor['meteor']}")

    predictions = [
        "It's a wonderful day", "I love dogs", "DataCamp has great AI courses",
        "Sunshine and flowers"
    ]
    references = [
        "What a wonderful day", "I love cats", "DataCamp has great AI courses",
        "Sunsets and flowers"
    ]

    exact_match = evaluate.load("exact_match")

    scenario3_results = exact_match.compute(references=references,
                                            predictions=predictions)
    print("Scenario 3:")
    print(scenario3_results)


if __name__ == '__main__':
    main()
