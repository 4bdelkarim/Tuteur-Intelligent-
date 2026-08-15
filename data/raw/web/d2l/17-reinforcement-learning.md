---
source_url: "https://d2l.ai/chapter_reinforcement-learning/index.html"
title: "17. Reinforcement Learning"
chapter: "17"
section_number: "17"
date: null
extractor: "d2l"
---

# 17. Reinforcement Learning

**Pratik Chaudhari** (*University of Pennsylvania and Amazon*), **Rasool
Fakoor** (*Amazon*), and **Kavosh Asadi** (*Amazon*)

Reinforcement Learning (RL) is a suite of techniques that allows us to
build machine learning systems that take decisions sequentially. For
example, a package containing new clothes that you purchased from an
online retailer arrives at your doorstep after a sequence of decisions,
e.g., the retailer finding the clothes in the warehouse closest to your
house, putting the clothes in a box, transporting the box via land or by
air, and delivering it to your house within the city. There are many
variables that affect the delivery of the package along the way, e.g.,
whether or not the clothes were available in the warehouse, how long it
took to transport the box, whether it arrived in your city before the
daily delivery truck left, etc. The key idea is that at each stage these
variables that we do not often control affect the entire sequence of
events in the future, e.g., if there were delays in packing the box in
the warehouse the retailer may need to send the package via air instead
of ground to ensure a timely delivery. Reinforcement Learning methods
allow us to take the appropriate action at each stage of a sequential
decision making problem in order to maximize some utility eventually,
e.g., the timely delivery of the package to you.

Such sequential decision making problems are seen in numerous other
places, e.g., while playing
[Go](https://en.wikipedia.org/wiki/Go_(game)) your current move
determines the next moves and the opponent’s moves are the variables
that you cannot control… a sequence of moves eventually determines
whether or not you win; the movies that Netflix recommends to you now
determine what you watch, whether you like the movie or not is unknown
to Netflix, eventually a sequence of movie recommendations determines
how satisfied you are with Netflix. Reinforcement learning is being used
today to develop effective solutions to these problems
(Mnih et al., 2013, Silver et al., 2016). The key
distinction between reinforcement learning and standard deep learning is
that in standard deep learning the prediction of a trained model on one
test datum does not affect the predictions on a future test datum; in
reinforcement learning decisions at future instants (in RL, decisions
are also called actions) are affected by what decisions were made in the
past.

In this chapter, we will develop the fundamentals of reinforcement
learning and obtain hands-on experience in implementing some popular
reinforcement learning methods. We will first develop a concept called a
Markov Decision Process (MDP) which allows us to think of such
sequential decision making problems. An algorithm called Value Iteration
will be our first insight into solving reinforcement learning problems
under the assumption that we know how the uncontrolled variables in an
MDP (in RL, these controlled variables are called the environment)
typically behave. Using the more general version of Value Iteration, an
algorithm called Q-Learning, we will be able to take appropriate actions
even when we do not necessarily have full knowledge of the environment.
We will then study how to use deep networks for reinforcement learning
problems by imitating the actions of an expert. And finally, we will
develop a reinforcement learning method that uses a deep network to take
actions in unknown environments. These techniques form the basis of more
advanced RL algorithms that are used today in a variety of real-world
applications, some of which we will point to in the chapter.

Fig. 17.1 Reinforcement Learning Structure

- 17.1. Markov Decision Process (MDP)
  - 17.1.1. Definition of an MDP
  - 17.1.2. Return and Discount Factor
  - 17.1.3. Discussion of the Markov Assumption
  - 17.1.4. Summary
  - 17.1.5. Exercises
- 17.2. Value Iteration
  - 17.2.1. Stochastic Policy
  - 17.2.2. Value Function
  - 17.2.3. Action-Value Function
  - 17.2.4. Optimal Stochastic Policy
  - 17.2.5. Principle of Dynamic Programming
  - 17.2.6. Value Iteration
  - 17.2.7. Policy Evaluation
  - 17.2.8. Implementation of Value Iteration
  - 17.2.9. Summary
  - 17.2.10. Exercises
- 17.3. Q-Learning
  - 17.3.1. The Q-Learning Algorithm
  - 17.3.2. An Optimization Problem Underlying Q-Learning
  - 17.3.3. Exploration in Q-Learning
  - 17.3.4. The “Self-correcting” Property of Q-Learning
  - 17.3.5. Implementation of Q-Learning
  - 17.3.6. Summary
  - 17.3.7. Exercises
