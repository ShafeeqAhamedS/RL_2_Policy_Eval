# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
<img width="650" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e7af87e7-fe73-47fa-8bea-2040b7645e44">

## POLICY EVALUATION FUNCTION

### Formula
<img width="450" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e663bd3d-fc85-41c3-9a5c-dffa57eae250">

### Program
```py
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    # Intialize 1st Iteration estimates of state-value function(V) to zero
    prev_V = np.zeros(len(P), dtype=np.float64)

    while True:
        # Intialize the current iteration estmates to zero
        V=np.zeros(len(P),dtype=np.float64)

        # Loop through all states
        for s in range(len(P)):

            # Loop throught action of our policy for every state
            for prob,next_state,reward,done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))

            # Check the difference between State Value Function between after each iteration, to see whetehr it have converged 
            if np.max(np.abs(prev_V-V))<theta:
                break

            # If not converged update the Value of previous estimation with current estimations
            prev_V=V.copy()
        return V
```

## OUTPUT:
### Policy 1
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/0a210271-47e9-4ea5-ab83-35053faaf4a5)
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/b3b933fd-60c8-48ec-b45b-07721b2d58bd)
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/c8ec140b-ac35-41cb-b068-d2e3f66fbc3b)
### Policy 2
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/c5863b33-0d56-45e6-9836-3b69f9515c4d)
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/f1875c9e-91fb-467c-ad63-df77c705ae0a)
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/0435f712-b2a4-44e8-ab64-a37fde842551)
### Comparison
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/4c85b228-5826-43a1-9733-44a99a57cb42)
</br>
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/4c0e961c-301e-4d9b-96cc-cdeb490f13c9)


## RESULT:
Thus, a Python program is developed to evaluate the given policy.
