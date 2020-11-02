#####################################################################################################
# Q-learning
#####################################################################################################

# install.packages("ggplot2")
# install.packages("vctrs")
library(ggplot2)

# If you do not see four arrows in line 16, then do the following:
# File/Reopen with Encoding/UTF-8

arrows <- c("↑", "→", "↓", "←")
action_deltas <- list(c(1,0), # up
                      c(0,1), # right
                      c(-1,0), # down
                      c(0,-1)) # left

vis_environment <- function(iterations=0, epsilon = 0.5, alpha = 0.1, gamma = 0.95, beta = 0){
  
  # Visualize an environment with rewards. 
  # Q-values for all actions are displayed on the edges of each tile.
  # The (greedy) policy for each state is also displayed.
  # 
  # Args:
  #   iterations, epsilon, alpha, gamma, beta (optional): for the figure title.
  #   reward_map (global variable): a HxW array containing the reward given at each state.
  #   q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
  #   H, W (global variables): environment dimensions.
  
  df <- expand.grid(x=1:H,y=1:W)
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,1],NA),df$x,df$y)
  df$val1 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,2],NA),df$x,df$y)
  df$val2 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,3],NA),df$x,df$y)
  df$val3 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,4],NA),df$x,df$y)
  df$val4 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) 
    ifelse(reward_map[x,y] == 0,arrows[GreedyPolicy(x,y)],reward_map[x,y]),df$x,df$y)
  df$val5 <- as.vector(foo)
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,max(q_table[x,y,]),
                                     ifelse(reward_map[x,y]<0,NA,reward_map[x,y])),df$x,df$y)
  df$val6 <- as.vector(foo)
  
  print(ggplot(df,aes(x = y,y = x)) +
          scale_fill_gradient(low = "white", high = "green", na.value = "red", name = "") +
          geom_tile(aes(fill=val6)) +
          geom_text(aes(label = val1),size = 4,nudge_y = .35,na.rm = TRUE) +
          geom_text(aes(label = val2),size = 4,nudge_x = .35,na.rm = TRUE) +
          geom_text(aes(label = val3),size = 4,nudge_y = -.35,na.rm = TRUE) +
          geom_text(aes(label = val4),size = 4,nudge_x = -.35,na.rm = TRUE) +
          geom_text(aes(label = val5),size = 10) +
          geom_tile(fill = 'transparent', colour = 'black') + 
          ggtitle(paste("Q-table after ",iterations," iterations\n",
                        "(epsilon = ",epsilon,", alpha = ",alpha,"gamma = ",gamma,", beta = ",beta,")")) +
          theme(plot.title = element_text(hjust = 0.5)) +
          scale_x_continuous(breaks = c(1:W),labels = c(1:W)) +
          scale_y_continuous(breaks = c(1:H),labels = c(1:H)))
  
}

GreedyPolicy <- function(x, y){
  
  # Get a greedy action for state (x,y) from q_table.
  #
  # Args:
  #   x, y: state coordinates.
  #   q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
  # 
  # Returns:
  #   An action, i.e. integer in {1,2,3,4}.
  
  # Your code here.
  maxUtility = -10000
  maxAction = 1
  #randomized order, if several max return the first one
  actions = sample(c(1,2,3,4))
  for(action in actions){
    qvalue = q_table[x,y,action]
    if (qvalue > maxUtility) {
      maxUtility = qvalue
      maxAction = action
    }
  }
  return(maxAction)
  
}

EpsilonGreedyPolicy <- function(x, y, epsilon){
  
  # Get an epsilon-greedy action for state (x,y) from q_table.
  #
  # Args:
  #   x, y: state coordinates.
  #   epsilon: probability of acting randomly.
  # 
  # Returns:
  #   An action, i.e. integer in {1,2,3,4}.
  
  # Your code here.

  maxAction = GreedyPolicy(x,y)
  # 1-epsilon prob of going by best action, else takes random
  if(rbinom(1,1,1-epsilon)==1){
    return(maxAction)
  }
    return(sample(1:4,1))
}

transition_model <- function(x, y, action, beta){
  
  # Computes the new state after given action is taken. The agent will follow the action 
  # with probability (1-beta) and slip to the right or left with probability beta/2 each.
  # 
  # Args:
  #   x, y: state coordinates.
  #   action: which action the agent takes (in {1,2,3,4}).
  #   beta: probability of the agent slipping to the side when trying to move.
  #   H, W (global variables): environment dimensions.
  # 
  # Returns:
  #   The new state after the action has been taken.
  
  delta <- sample(-1:1, size = 1, prob = c(0.5*beta,1-beta,0.5*beta))
  final_action <- ((action + delta + 3) %% 4) + 1
  foo <- c(x,y) + unlist(action_deltas[final_action])
  foo <- pmax(c(1,1),pmin(foo,c(H,W)))
  
  return (foo)
}

q_learning_old <-
  function(start_state,
           epsilon = 0.5,
           alpha = 0.1,
           gamma = 0.95,
           beta = 0) {
    # Perform one episode of Q-learning. The agent should move around in the
    # environment using the given transition model and update the Q-table.
    # The episode ends when the agent reaches a terminal state.
    #
    # Args:
    # start_state: array with two entries, describing the starting position of the agent.
    # epsilon (optional): probability of acting greedily.
    # alpha (optional): learning rate.
    # gamma (optional): discount factor.
    # beta (optional): slipping factor.
    # reward_map (global variable): a HxW array containing the reward given at each state.
    # q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
    #
    # Returns:
    # reward: reward received in the episode.
    # correction: sum of the temporal difference correction terms over the episode.
    # q_table (global variable): Recall that R passes arguments by value. So, q_table being
    # a global variable can be modified with the superassigment operator <<-.
    
    # Your code here.
    state = start_state
    episodeCorrection = 0
    repeat {
      # Follow policy, execute action, get reward.
      action = EpsilonGreedyPolicy(state[1], state[2], epsilon)
      newState = transition_model(state[1], state[2], action, beta)
      reward = reward_map[newState[1], newState[2]]
      
      maxActionNewState = max(q_table[newState[1], newState[2], ])
      # Q-table update
      temporalDifference = (reward + gamma * max(q_table[newState[1], newState[2], ])
                            - q_table[state[1], state[2], action])
      
      q_table[state[1], state[2], action] <<-
        (q_table[state[1], state[2], action] + alpha * (temporalDifference))
      state = newState
      episodeCorrection = episodeCorrection + temporalDifference
      if (reward != -1) {
        # End episode.
        return (c(reward, episodeCorrection))
      }
    }
    
  }

q_learning_old_test <-
  function(start_state,
           epsilon = 0.5,
           alpha = 0.1,
           gamma = 0.95,
           beta = 0) {
    # Perform one episode of Q-learning. The agent should move around in the
    # environment using the given transition model and update the Q-table.
    # The episode ends when the agent reaches a terminal state.
    #
    # Args:
    # start_state: array with two entries, describing the starting position of the agent.
    # epsilon (optional): probability of acting greedily.
    # alpha (optional): learning rate.
    # gamma (optional): discount factor.
    # beta (optional): slipping factor.
    # reward_map (global variable): a HxW array containing the reward given at each state.
    # q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
    #
    # Returns:
    # reward: reward received in the episode.
    # correction: sum of the temporal difference correction terms over the episode.
    # q_table (global variable): Recall that R passes arguments by value. So, q_table being
    # a global variable can be modified with the superassigment operator <<-.
    
    # Your code here.
    state = start_state
    episodeCorrection = 0
    repeat {
      # Follow policy, execute action, get reward.
      action = GreedyPolicy(state[1], state[2])
      newState = transition_model(state[1], state[2], action, beta)
      reward = reward_map[newState[1], newState[2]]
      
      # Q-table update
     
      if (reward != -1) {
        # End episode.
        return (c(reward, episodeCorrection))
      }
    }
    
  }

q_learning <-
  function(start_state,
           epsilon = 0.5,
           alpha = 0.1,
           gamma = 0.95,
           beta = 0) {
    # Perform one episode of Q-learning. The agent should move around in the
    # environment using the given transition model and update the Q-table.
    # The episode ends when the agent reaches a terminal state.
    #
    # Args:
    # start_state: array with two entries, describing the starting position of the agent.
    # epsilon (optional): probability of acting greedily.
    # alpha (optional): learning rate.
    # gamma (optional): discount factor.
    # beta (optional): slipping factor.
    # reward_map (global variable): a HxW array containing the reward given at each state.
    # q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
    #
    # Returns:
    # reward: reward received in the episode.
    # correction: sum of the temporal difference correction terms over the episode.
    # q_table (global variable): Recall that R passes arguments by value. So, q_table being
    # a global variable can be modified with the superassigment operator <<-.
    
    # Your code here.
    state = start_state
    episodeCorrection = 0
    nextAction = EpsilonGreedyPolicy(state[1], state[2], epsilon)
    repeat {
      # Follow policy, execute action, get reward.
      action = nextAction
      newState = transition_model(state[1], state[2], action, beta)
      reward = reward_map[newState[1], newState[2]]
      
      nextAction = EpsilonGreedyPolicy(newState[1], newState[2], epsilon)
      
      # Q-table update
      temporalDifference = (reward + gamma * q_table[newState[1], newState[2], nextAction]
                            - q_table[state[1], state[2], action])
      
      q_table[state[1], state[2], action] <<-
        (q_table[state[1], state[2], action] + alpha * (temporalDifference))
      state = newState
      episodeCorrection = episodeCorrection + temporalDifference
      if (reward != -1) {
        # End episode.
        return (c(reward, episodeCorrection))
      }
    }
    
  }

q_learning_test <-
  function(start_state,
           epsilon = 0.5,
           alpha = 0.1,
           gamma = 0.95,
           beta = 0) {
    # Perform one episode of Q-learning. The agent should move around in the
    # environment using the given transition model and update the Q-table.
    # The episode ends when the agent reaches a terminal state.
    #
    # Args:
    # start_state: array with two entries, describing the starting position of the agent.
    # epsilon (optional): probability of acting greedily.
    # alpha (optional): learning rate.
    # gamma (optional): discount factor.
    # beta (optional): slipping factor.
    # reward_map (global variable): a HxW array containing the reward given at each state.
    # q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
    #
    # Returns:
    # reward: reward received in the episode.
    # correction: sum of the temporal difference correction terms over the episode.
    # q_table (global variable): Recall that R passes arguments by value. So, q_table being
    # a global variable can be modified with the superassigment operator <<-.
    
    # Your code here.
    state = start_state
    episodeCorrection = 0
    # in test greedy is done, not epsilongreedy
    nextAction = GreedyPolicy(state[1], state[2])
    repeat {
      # Follow policy, execute action, get reward.
      action = nextAction
      newState = transition_model(state[1], state[2], action, beta)
      reward = reward_map[newState[1], newState[2]]
      # in test greedy is done, not epsilongreedy
      nextAction = GreedyPolicy(newState[1], newState[2])
      state = newState
      if (reward == -10 || reward == 10) {
        # End episode.
        return (c(reward, episodeCorrection))
      }
    }
    
  }

#####################################################################################################
# Q-Learning Environments
#####################################################################################################

# Environment A (learning)

H <- 5
W <- 7

reward_map <- matrix(0, nrow = H, ncol = W)
reward_map[3,6] <- 10
reward_map[2:4,3] <- -1

q_table <- array(0,dim = c(H,W,4))

vis_environment()

for(i in 1:10000){
  foo <- q_learning(start_state = c(3,1))
  
  if(any(i==c(10,100,1000,10000)))
    vis_environment(i)
}


# Environment B (the effect of epsilon and gamma)

H <- 7
W <- 8

reward_map <- matrix(0, nrow = H, ncol = W)
reward_map[1,] <- -1
reward_map[7,] <- -1
reward_map[4,5] <- 5
reward_map[4,8] <- 10

q_table <- array(0,dim = c(H,W,4))

vis_environment()

MovingAverage <- function(x, n){
  
  cx <- c(0,cumsum(x))
  rsum <- (cx[(n+1):length(cx)] - cx[1:(length(cx) - n)]) / n
  
  return (rsum)
}

for(j in c(0.5,0.75,0.95)){
  q_table <- array(0,dim = c(H,W,4))
  reward <- NULL
  correction <- NULL
  
  for(i in 1:30000){
    foo <- q_learning(gamma = j, start_state = c(4,1))
    reward <- c(reward,foo[1])
    correction <- c(correction,foo[2])
  }
  
  vis_environment(i, gamma = j)
  plot(MovingAverage(reward,100),type = "l")
  plot(MovingAverage(correction,100),type = "l")
}

for(j in c(0.5,0.75,0.95)){
  q_table <- array(0,dim = c(H,W,4))
  reward <- NULL
  correction <- NULL
  
  for(i in 1:30000){
    foo <- q_learning(epsilon = 0.1, gamma = j, start_state = c(4,1))
    reward <- c(reward,foo[1])
    correction <- c(correction,foo[2])
  }
  
  vis_environment(i, epsilon = 0.1, gamma = j)
  plot(MovingAverage(reward,100),type = "l")
  plot(MovingAverage(correction,100),type = "l")
}

# Environment C (the effect of beta).

H <- 3
W <- 6

MovingAverage <- function(x, n){
  
  cx <- c(0,cumsum(x))
  rsum <- (cx[(n+1):length(cx)] - cx[1:(length(cx) - n)]) / n
  
  return (rsum)
}


reward_map <- matrix(-1, nrow = H, ncol = W)
reward_map[1,2:5] <- -10
reward_map[1,6] <- 10

q_table <- array(0,dim = c(H,W,4))

vis_environment()
reward1 <- NULL
reward2 <- NULL


  for(i in 1:5000){
    foo <- q_learning_test(gamma = 1, beta = 0, start_state = c(1,1))
    foo2 <- q_learning_old_test(gamma = 1, beta = 0, start_state = c(1,1))
    reward1 <- c(reward1,foo[1])
    reward2 <- c(reward2,foo[2])
    
  }
  vis_environment(5000, gamma = 1, beta = 0)
  plot(MovingAverage(reward1,100),type = "l", main="Original Q-learning")
  plot(MovingAverage(reward2,100),type = "l",main="New Q-learning")
  
  
