#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:19:53 2019

@author: radwaelaraby
"""

from env_CatchPigs import EnvCatchPigs
import random
import numpy as np 

LEARNING_RATE = 0.4
DEGRADING_FACTOR = 0.8
GREEDY_EPISON = 0.5
START = 500
NO_OF_ITERATIONS = 501
TD_LAMDA = 0.9

LEFT_ORI = '0'
UP_ORI = '1'
RIGHT_ORI = '2'
DOWN_ORI = '3'

ORIENTATIONS = [LEFT_ORI, UP_ORI, RIGHT_ORI, DOWN_ORI]

TURN_LEFT = 0
TURN_RIGHT = 1
MOVE_FORWARD = 2
CATCH = 3 
WAIT = 4 

ACTIONS = [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, CATCH, WAIT]
ACTION_SIZE = np.array(ACTIONS).size

A1_Q = {}
A2_Q = {}
A1_ACCORDING_TO_A2 = {}
A2_ACCORDING_TO_A1 = {}

POSSIBLE_Xs = np.arange(2, 5, 1)
POSSIBLE_Ys = np.arange(2, 5, 1)

possible_places = []
pig_places = []
for x in POSSIBLE_Xs: 
    for y in POSSIBLE_Ys: 
        p_key = str(x) + '_' + str(y)
        pig_places.append(p_key)            
        for p in ORIENTATIONS:
            a_key = str(x) + '_' + str(y) + '_' + str(p)
            possible_places.append(a_key)

A1_A2_places = []
for a1_place in possible_places: 
    for a2_place in possible_places: 
        if (a1_place[:3] == a2_place[:3]):
            continue
        key = str(a1_place) + '_' + str(a2_place)
        A1_A2_places.append(key)

for i in A1_A2_places:
    for j in pig_places:
        if (i[:3] == j):
            continue
        s_key = str(i) + '_' + str(j)
        A1_Q[s_key] = np.zeros((ACTION_SIZE, ACTION_SIZE))
        A2_Q[s_key] = np.zeros((ACTION_SIZE, ACTION_SIZE))
        A1_ACCORDING_TO_A2[s_key] = np.zeros((ACTION_SIZE, 1))
        A2_ACCORDING_TO_A1[s_key] = np.zeros((ACTION_SIZE, 1))



def playOneGame():
    env = EnvCatchPigs(7, True)
    
    done = False
    A1_traces = {}
    A2_traces = {}    
    
    A1_px, A1_py = env.agt1_pos
    A2_px, A2_py = env.agt2_pos
    A1_ori = env.agt1_ori
    A2_ori = env.agt2_ori
    P_px, P_py = env.pig_pos        
        
    A1_place = str(A1_px) + '_' + str(A1_py) + '_' + str(A1_ori)
    A2_place = str(A2_px) + '_' + str(A2_py) + '_' + str(A2_ori)
    pig_place = str(P_px) + '_' + str(P_py)
    
    A1_s = A1_place + '_' + A2_place + '_' + pig_place
    A2_s = A2_place + '_' + A1_place + '_' + pig_place

    A1_a = random.choice(ACTIONS)
    A2_a = random.choice(ACTIONS)
    
    env.render()
    
    while done == False:
                
        A1_ACCORDING_TO_A2[A2_s][A1_a] += 1
        A2_ACCORDING_TO_A1[A1_s][A2_a] += 1
         
        action_profile = [A1_a, A2_a]

        r, done = env.step(action_profile)
        
        env.render()
        
        A1_r = r[0]
        A2_r = r[1]
        
        A1_px, A1_py = env.agt1_pos
        A2_px, A2_py = env.agt2_pos
        A1_ori = env.agt1_ori
        A2_ori = env.agt2_ori
        P_px, P_py = env.pig_pos
            
        A1_place = str(A1_px) + '_' + str(A1_py) + '_' + str(A1_ori)
        A2_place = str(A2_px) + '_' + str(A2_py) + '_' + str(A2_ori)
        pig_place = str(P_px) + '_' + str(P_py)
        
        A1_s_ = A1_place + '_' + A2_place + '_' + pig_place
        A2_s_ = A2_place + '_' + A1_place + '_' + pig_place
        
        if (random.random() > GREEDY_EPISON): 
            A1_Q_expected = np.dot(A1_Q[A1_s_], A2_ACCORDING_TO_A1[A1_s_])
            if (np.sum(A2_ACCORDING_TO_A1[A1_s_]) > 0):
                A1_Q_expected /= np.sum(A2_ACCORDING_TO_A1[A1_s_])
            A1_a_ = np.random.choice(np.flatnonzero(A1_Q_expected == A1_Q_expected.max()))
            
        else: 
            A1_a_ = random.choice(ACTIONS)


        if (random.random() > GREEDY_EPISON): 
            A2_Q_expected = np.dot(A2_Q[A2_s_], A1_ACCORDING_TO_A2[A2_s_])
            if (np.sum(A1_ACCORDING_TO_A2[A2_s_]) > 0):
                A2_Q_expected /= np.sum(A1_ACCORDING_TO_A2[A2_s_])
            A2_a_ = np.random.choice(np.flatnonzero(A2_Q_expected == A2_Q_expected.max()))
            
        else: 
            A2_a_ = random.choice(ACTIONS)
 
        A1_td_err = (A1_r + (DEGRADING_FACTOR * A1_Q[A1_s_][A1_a_][A2_a_])) - A1_Q[A1_s][A1_a][A2_a]
        A2_td_err = (A2_r + (DEGRADING_FACTOR * A2_Q[A2_s_][A2_a_][A1_a_])) - A2_Q[A2_s][A2_a][A1_a]
        
        A1_combination_key = A1_s + ':' + str(A1_a) + ':' + str(A2_a)
        if ((A1_combination_key in A1_traces) == False):
            A1_traces[A1_combination_key] = 0.0

        A2_combination_key = A2_s + ':' + str(A2_a) + ':' + str(A1_a)
        if ((A2_combination_key in A2_traces) == False):
            A2_traces[A2_combination_key] = 0.0
            
        for A1_combination_key, A1_trace_value in A1_traces.items():
            m_s, m_a, o_a = A1_combination_key.split(':')
            o_a = int(o_a)
            m_a = int(m_a)
            if (A1_s == m_s and A2_a == o_a and A1_a == m_a):     
                A1_traces[A1_combination_key] = (TD_LAMDA * A1_trace_value) + 1
            else: 
                A1_traces[A1_combination_key] = (TD_LAMDA * A1_trace_value)
                
            A1_Q[m_s][m_a][o_a] += LEARNING_RATE * A1_traces[A1_combination_key] * A1_td_err
            
        for A2_combination_key, A2_trace_value in A2_traces.items():
            m_s, m_a, o_a = A2_combination_key.split(':')
            o_a = int(o_a)
            m_a = int(m_a)

            if (A2_s == m_s and A1_a == o_a and A2_a == m_a):     
                A2_traces[A2_combination_key] = (TD_LAMDA * A2_trace_value) + 1
            else: 
                A2_traces[A2_combination_key] = (TD_LAMDA * A2_trace_value)

            A2_Q[m_s][m_a][o_a] += LEARNING_RATE * A2_traces[A2_combination_key] * A2_td_err        
                    
        A1_s = A1_s_
        A2_s = A2_s_

        A1_a = A1_a_
        A2_a = A2_a_
            
for i in range(START, NO_OF_ITERATIONS):    
    playOneGame()
