import numpy as np
import matplotlib.pyplot as plt


def softmax_l_layer(prev_output):
  max_logit = np.max(prev_output)
  prev_output = prev_output - max_logit # prevents numerical instability
  deno = np.sum(np.exp(prev_output))
  # output = np.empty(len(prev_output)) # np.empty > np.zeros, for speed
  output = (np.exp(prev_output)/ deno)
  
  return output



#EVEN BETTER and VECTORISED
def softmax_l_layer_mini(prev_output):
  prev_output = prev_output - np.max(prev_output) # prevents numerical instability
  exps= np.exp(prev_output)
  
  return exps / np.sum(exps)


