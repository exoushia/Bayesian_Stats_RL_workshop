"""
This code file contains a Bayes Theorem test case
"""

"""
===================
Bayes Rule
Bayes’ theorem (alternatively Bayes’ law or Bayes’ rule) describes the probability of an event, 
based on prior knowledge of conditions that might be related to the event. 
For example, if a disease is related to age, then, using Bayes’ theorem, 
a person's age can be used to more accurately assess the probability that they have the disease, 
compared to the assessment of the probability of disease made without knowledge of the person’s age.

===================

Drug screening example
Suppose that a test for using a particular drug is 97% sensitive and 95% specific.
That is, the test will produce 97% true positive results for drug users and 95% true negative results for non-drug users.
Suppose that 0.5% of the general population are users of the drug. 
What is the probability that a randomly selected individual with a positive test is a drug user?

=========================

Goal of this code file :
We will write a custom function which accepts the test capabilities 
and the prior knowledge of drug user percentage as input 
and produces the output probability of a test-taker being an user based on a positive result.

======================

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
output_directory = 'Output_directory'

def is_user_drug_user(threshold=0.5, sensitivity=0.97, specificity=0.95, prevelance_drug_user_in_population=0.01, verbose=True):
    """
    custom function which accepts the test capabilities and the prior knowledge of drug user percentage
    as input and determines whether a user is a drug user
    """
    p_user = prevelance_drug_user_in_population
    p_non_user = 1 - prevelance_drug_user_in_population
    p_pos_user = sensitivity  # p(user=drug_user | test +ve)
    p_neg_user = specificity
    p_pos_non_user = 1 - specificity  # p(user=drug_user | test ive)

    num = p_pos_user * p_user
    den = p_pos_user * p_user + p_pos_non_user * p_non_user

    prob = num / den

    if verbose:
        if prob > threshold:
            print("The test-taker could be an user")
        else:
            print("The test-taker may not be an user")

    return prob

if __name__ == '__main__':
    probability_drug_user = is_user_drug_user(threshold=0.5,
                                              sensitivity=0.97,
                                              specificity=0.95,
                                              prevelance_drug_user_in_population=0.005)

    print("Probability of the test-taker being a drug user is:", round(probability_drug_user, 3))

    print("\n=======")
    print("Bayesian inference: Ability to use prior knowledge to update beliefs about parameters")
    p1 = is_user_drug_user(threshold=0.5,sensitivity=0.97,specificity=0.95,prevelance_drug_user_in_population=0.005)
    print("Probability of the test-taker being a drug user, in the first round of test, is:",round(p1,3))
    print("-------")
    p2 = is_user_drug_user(threshold=0.5,sensitivity=0.97,specificity=0.95,prevelance_drug_user_in_population=p1)
    print("Probability of the test-taker being a drug user, in the second round of test, is:",round(p2,3))
    print("-------")
    p3 = is_user_drug_user(threshold=0.5,sensitivity=0.97,specificity=0.95,prevelance_drug_user_in_population=p2)
    print("Probability of the test-taker being a drug user, in the third round of test, is:",round(p3,3))
    print("=======")
    print("With three consecutive tests, we are 97.3% confident about catching a true drug user, with the same test "
          "capabilities.")

    ps = []
    pres = []
    for pre in [i*0.001 for i in range(1, 51, 2)]:
        pres.append(pre*100)
        p = is_user_drug_user(threshold=0.5,
                              sensitivity=0.97,
                              specificity=0.95,
                              prevelance_drug_user_in_population=pre,
                              verbose=False)
        ps.append(p)

    plt.figure(figsize=(10,5))
    plt.title("Probability of user with prevalence rate",fontsize=15)
    plt.plot(pres, ps, color='k', marker='o', markersize=8)
    plt.grid(True)
    plt.xlabel("Prevelance (percentage)",fontsize=14)
    plt.ylabel("Probability of being an user",fontsize=14)
    plt.xticks([i*0.25 for i in range(1,21)],fontsize=12,rotation=45)
    plt.yticks(fontsize=12)
    plt.savefig(f"{output_directory}/drug_user_ex_bayes.png")
    plt.close()
