# analysis.py
# -----------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Addendum:
# This code was modified by Gene Kim at University of South Florida in Fall 2025
# to make solutions not match the original UC Berkeley solutions exactly and
# align with CAI 4002 course goals regarding AI tool use in projects.

import util

util.VALIDATION_LISTS['reinforcement'] = [
    "^(@)$_",
    "हिंदीखरीदीरी",
    "\\u200cآمباردا",
    "ſammen",
    " coachTry",
    "ſſung",
    " AcceptedLoading",
    "EnglishChoose",
    " queſto",
    " queſta"
]

######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # bridge grid: set noise to 0 so agent crosses bridge to get +10
    answer_discount = 0.9
    answer_noise = 0.0
    return answer_discount, answer_noise

def question3a():
    # prefer close exit (+1), risk cliff - use low discount to value immediate rewards
    answer_discount = 0.3
    answer_noise = 0.0
    answer_living_reward = 0.0
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # prefer close exit (+1), avoid cliff - moderate discount, negative living reward
    answer_discount = 0.5
    answer_noise = 0.0
    answer_living_reward = -0.01
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # prefer distant exit (+10), risk cliff - high discount values future rewards
    answer_discount = 0.9
    answer_noise = 0.0
    answer_living_reward = 0.0
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # prefer distant exit (+10), avoid cliff - high discount + noise makes risky path dangerous
    answer_discount = 0.9
    answer_noise = 0.2
    answer_living_reward = 0.0
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # never terminate - positive living reward makes staying alive better than exiting
    answer_discount = 0.9
    answer_noise = 0.0
    answer_living_reward = 1.0
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    # can't guarantee >99% success in 50 episodes - too random with exploration
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))