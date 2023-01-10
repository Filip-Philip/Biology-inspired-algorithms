from math import floor
from sys import argv
import numpy as np
import random
import matplotlib.pyplot as plt

if len(argv) != 6:
    print("Not enough arguments")
    exit(1)

STUDENT_COUNT = int(argv[1])  # 10
ITERATIONS = int(argv[2])  # 500
POPULATION = int(argv[3])  # 100
FIT_SOLUTIONS = int(argv[4])  # 20
MUTATION_ACCEPTANCE = float(argv[5])  # 0.05
SUBJECTS = [
    [11, 12, 14],
    [10, 12, 16],
    [11, 12, 18],
    [31, 32, 34],
    [30, 32, 36],
    [31, 32, 38],
    [51, 52, 54],
    [50, 52, 56],
    [51, 52, 58],
]
CLASS_DURATION = 1


def are_colliding(subject1, date1, subject2, date2):
    return (
        np.abs(SUBJECTS[subject1][date1] - SUBJECTS[subject2][date2]) < CLASS_DURATION
    )


def child(sol1: np.ndarray, sol2: np.ndarray):
    c = np.zeros_like(sol1)

    for st in range(sol1.shape[0]):
        for sep in range(1, sol1.shape[1]):
            sep_legal = True
            for s1 in range(sep):
                for s2 in range(sep, sol1.shape[1]):
                    coll = are_colliding(s1, sol1[st, s1], s2, sol2[st, s2])
                    if coll:
                        sep_legal = False
                        break

                if not sep_legal:
                    break

            if sep_legal:
                c[st, :sep] = sol1[st, :sep]
                c[st, sep:] = sol2[st, sep:]
                break
        else:
            c[st, :] = sol1[st, :]

    return c


def generate_random_schedule(students):
    schedule = np.zeros((students, len(SUBJECTS)), dtype=int) - 1

    for st in range(students):
        for sub in range(len(SUBJECTS)):
            dates = list(range(len(SUBJECTS[sub])))
            random.shuffle(dates)

            for date in dates:
                date_valid = True
                for prev_sub in range(sub):
                    coll = are_colliding(sub, date, prev_sub, schedule[st, prev_sub])
                    if coll:
                        date_valid = False
                        break

                if date_valid:
                    schedule[st, sub] = date
                    break

    return schedule


def mutate(schedule: np.ndarray):
    number_of_students, number_of_subjects = schedule.shape
    if random.random() < MUTATION_ACCEPTANCE:
        percentage_to_swap = random.random()
        student1 = random.randint(0, number_of_students - 1)
        student2 = random.randint(0, number_of_students - 1)
        for i in range(floor(number_of_subjects * percentage_to_swap)):
            if not are_colliding(i, schedule[student1, i], i, schedule[student2, i]):
                schedule[student1, i], schedule[student2, i] = (
                    schedule[student2, i],
                    schedule[student1, i],
                )

    return schedule


def student_fitness(dates: np.ndarray):
    SCHOOL_DAYS = 5
    HOURS_IN_DAY = 24

    subjects_in_day = [[] for _ in range(SCHOOL_DAYS)]

    for i in range(dates.shape[0]):
        d = SUBJECTS[i][dates[i]]
        subjects_in_day[d // HOURS_IN_DAY].append(d)
    for subjects in subjects_in_day:
        subjects.sort()

    def day_fitness(hours):
        n = len(hours)
        breaks_time = 0
        for i in range(n - 1):
            breaks_time += hours[i + 1] - hours[i] - 1

        if n == 0 and breaks_time == 0:
            return 1
        return n / (n + breaks_time)

    return sum(map(day_fitness, subjects_in_day)) ** (1 / 2)


def overflow_fitness(schedule: np.ndarray):
    GROUP_MAX_COUNT = floor(STUDENT_COUNT * 1.5 / len(SUBJECTS[0]))
    group_count = np.array(SUBJECTS) * 0

    for st in range(schedule.shape[0]):
        for sub in range(schedule.shape[1]):
            group_count[sub, schedule[st, sub]] += 1

    students_over = np.clip(group_count.flatten() - GROUP_MAX_COUNT, 0, 100)

    return -sum(np.exp(students_over) - 1)


def fitness_function(schedule: np.ndarray):
    return sum(
        [student_fitness(schedule[s, :]) for s in range(schedule.shape[0])]
    ) + overflow_fitness(schedule)


plt.clf()

solutions = [generate_random_schedule(STUDENT_COUNT) for _ in range(POPULATION)]

scores = [0] * (ITERATIONS + 1)
solutions_scores = np.array(list(map(fitness_function, solutions)))
scores[0] = np.max(solutions_scores)

file_content = "i, score\n"

for i in range(ITERATIONS):
    if i / ITERATIONS * 100 % 10 == 0:
        print(f"{(i / ITERATIONS) * 100:.3}%")

    solutions.sort(key=fitness_function, reverse=True)

    weights = np.array(list(map(fitness_function, solutions)))[:FIT_SOLUTIONS]
    weights = weights / sum(weights)

    def get_fit_sol():
        return random.choices(solutions[:FIT_SOLUTIONS], weights, k=1)[0]

    solutions = [mutate(child(get_fit_sol(), get_fit_sol())) for _ in range(POPULATION)]

    solutions_scores = np.array(list(map(fitness_function, solutions)))
    scores[i + 1] = np.max(solutions_scores)
    file_content += f"{i}, {scores[i + 1]}\n"


name = f"SC{STUDENT_COUNT}_IT{ITERATIONS}_P{POPULATION}_F{FIT_SOLUTIONS}_M{MUTATION_ACCEPTANCE}"

plt.plot(scores)
plt.savefig(
    f"plot_{name}.pdf",
    bbox_inches="tight",
)

with open(f"data_{name}.txt", "w", encoding="utf8") as f:
    f.write(file_content)
