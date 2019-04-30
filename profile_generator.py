'''
	File: 	profile_generator.py
	Author:	Nicholas Mattei (nicholas.mattei@nicta.com.au)
	Date:	Sept 11, 2013
			November 6th, 2013
			July 30th, 2014

  * Copyright (c) 2014, Nicholas Mattei and NICTA
  * All rights reserved.
  *
  * Developed by: Nicholas Mattei
  *               NICTA
  *               http://www.nickmattei.net
  *               http://www.preflib.org
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *     * Redistributions of source code must retain the above copyright
  *       notice, this list of conditions and the following disclaimer.
  *     * Redistributions in binary form must reproduce the above copyright
  *       notice, this list of conditions and the following disclaimer in the
  *       documentation and/or other materials provided with the distribution.
  *     * Neither the name of NICTA nor the
  *       names of its contributors may be used to endorse or promote products
  *       derived from this software without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY NICTA ''AS IS'' AND ANY
  * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL NICTA BE LIABLE FOR ANY
  * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.	
	

About
--------------------
	This file generates voting profiles according to a given distribution.
	It requires PreflibUtils to work properly.

NOTE
--------------------
	This is a heavily modified version of this file which does not
	require the use of PreflibUtils.
		
'''
import random
import itertools
import math
import copy
import argparse
import sys
import collections
from scipy import stats
import numpy as np

_DEBUG = False

def generate_approx_m_regular_assignment(agents, m, clusters={}, randomize=True):
  """
  Creates an "approximatly" m regular reviewing assignment
  for all agents subject to the restriction that no one 
  reviews themselves or anyone in their cluster.  This is
  approximate as it will balance as much as possible
  but not be exact.  We attempt to do this in a cyclical manner;
  this algorithm will fail if each person is asked to review
  too many people or the clusters are not balanced.

  Parameters
  -----------
  agents: array like
  	list of agents to be reviewed.

  m: integer
  	Number of elements that each agent should review.  This does
  	not mean the number of reviews that each agent will recieve.

  clusters: dict
    A mapping from integer ---> [agents] where each agent in an 
    partition together.  Agents should not review other agents 
    in their own partition.  Also, partitions must be disjoint.

  Returns
  -----------
  assignment: dict
  	A dict from agent --> [agents] which is a list of lenght m
  	of agents not in the partition of agent i.

  Notes
  -----------
  """
  #Verify the clusters...
  if clusters != {}:
  	if _DEBUG: print("\nClusters:\n" + str(clusters))
  	#Ensure that the partitions don't overlap.
  	agent_set = list(itertools.chain(*clusters.values()))
  	if len(agent_set) != len(set(agent_set)):
  		print("clustering contains duplicates in different clusters")
  		return 0
  else:
  	#Make everone their own cluster if we don't have a clustering.
  	clusters = {i:[agents[i]] for i in range(len(agents))}
  	if _DEBUG: print("\nClusters:\n" + str(clusters))

  # Do a check here -- if, for every cluster, the number of agents
  # outside is < m fail, can't review 2x.
  if any([m > len(agents) - len(ci) for k,ci in clusters.items()]):
    print("m is larger than N - Ci for some Ci, duplicate review required.")
    return 0

  cluster_assignment = {}
  # Shuffling a dequq scales as n^2, faster to copy..
  t = list(clusters.keys())
  if randomize:
    random.shuffle(t)
  cluster_deq = collections.deque(t)

  for j in clusters.keys():
    current = cluster_deq.popleft()
    c_list = list(cluster_deq)
    assn = c_list*math.ceil(m*len(clusters[current]) / len(c_list))
    assn = assn[:m*len(clusters[current])]
    cluster_assignment[current] = assn
    cluster_deq.append(current)
  	
  if _DEBUG: print("Cluster Assignment: " + str(cluster_assignment))

  agent_to_clusters = {k:[] for k in agents}
  # Iterate over a cluster --> cluster assignment and convert it
  # to a agent --> cluster assignement using a canoical ordering
  # for the agents in each cluster on the RHS.
  for c_cluster in cluster_assignment.keys():
    c_agents = copy.copy(clusters[c_cluster])
    if randomize:
      random.shuffle(c_agents)
    targets = cluster_assignment[c_cluster]
    for c_a in c_agents:
      agent_to_clusters[c_a] = targets[:m]
      targets = targets[m:]

  if _DEBUG: print("Agents to Clusters: " + str(agent_to_clusters))

  # Convert the RHS.  For every agent, replace a agent --> cluster
  # assignment to a agent --> agent assignment.  Use a canoical 
  # ordering of the agents on the RHS randomized if necessary.

  agent_assignment = {k:[] for k in agents}
  target_order = {}
  # Build RHS ordering.
  for k,v in clusters.items():
    t = copy.copy(v)
    if randomize:
      random.shuffle(t)
    target_order[k] = collections.deque(t)

  for a, t in agent_to_clusters.items():
    # For each target cluster
    for cc in t:
      #Assign the front of the list and then rotate it 
      agent_assignment[a].append(target_order[cc][0])
      target_order[cc].rotate(-1)

  # Post check for duplicates..
  for k,v in agent_assignment.items():
    if len(v) != len(set(v)):
      print("Double review assignment: ", str(k), " :: ", str(v))
    if len(v) != m:
      print("Error in assignment, agent ", str(k), " has less than m reviews ", str(v))
  if _DEBUG: print("Agent to Agent: " + str(agent_assignment))

  return agent_assignment

def generate_mallows_mixture_profile(voters, candidates, distribution, reference_rankings, phis):
  """
  For every voter generates an ordinal preference profile over the 
  set of candidates according to a mallows mixture model 
  parameterized by a distribution over mallows models which 
  consist of a reference ranking and a dispersion parameter (phi)

  Parameters
  -----------
  voters: array like

  candidates: array like

  distribution: array like
    A distribution over discrete mallows models, must sum to 1.0.

  reference_rankings: array like
    Same lenght as distribution, must be a full ranking over candidates.

  phis: array like of floats
    Same length as distribtion, must be between 0.0 and 1.0 and describes
    the dispersion parameter for the indexed model.

  Returns
  -----------
  profile: dict
    A mapping from voters ---> [ranking] where the complete element
    set of candidats is ordered in ranking with ranking[0] being 
    the most prefered element.

  Notes
  -----------
  Generate a Mallows model with the various mixing parameters passed.
  Distribution should be a mix over the models that sums to ~1.0.
  phis is an array len(phis) = len(mix) = len(refs) that is the phi for the particular models

  Method mostly implemented from Lu, Tyler, and Craig Boutilier. 
  "Learning Mallows models with pairwise preferences." 
  Proceedings of the 28th International Conference on Machine Learning (
  ICML-11). 2011.

  """
  if not np.isclose(sum(distribution), 1.0):
    print("Mallows distribution isn't close to 1.0")
    return 0
  if len(distribution) != len(reference_rankings) or len(reference_rankings) != len(phis):
    print("Insufficient number of parameters for mallows mix.")
    return 0

  # Using Numpy discrete and precomputing the models has better speed for large N.
  # Using RVS in this way seems to scale much better.
  #       100, 1000, 2000
  # draw   <2,   94,   724s
  # RVS    <2,   92,   356s
  # Numpy RV version...


  # Define the RVS for the draw over distros...
  model_rvs = stats.rv_discrete(values=(list(range(len(distribution))),distribution))

  insertion_rvs = {}
  # For each of the models we need to precompute the insertion distributions and
  # associated RVs.
  for m,p in enumerate(phis):
    # Returns an insertion probability vector  according to phi.
    # this is the probability that ref[i-1] should be inserted into position []
    insertion_distribution = compute_mallows_insertion_distribution(len(candidates), p)
  
    # Make a set of RV's for each element.
    model_insertion_rvs = {}
    for i in insertion_distribution.keys():
      model_insertion_rvs[i] = stats.rv_discrete(values=(list(range(len(insertion_distribution[i]))), insertion_distribution[i]))
    insertion_rvs[m] = model_insertion_rvs

  profile = {}
  for c_voter in voters:
    # Draw a model.
    c_model = model_rvs.rvs()
    # Will generate a list (strict order) over the elements of candidates.
    profile[c_voter] = []
    for i,c in enumerate(reference_rankings[c_model]):
      profile[c_voter].insert(insertion_rvs[c_model][i].rvs(), c)

  return profile

def compute_mallows_insertion_distribution(length, phi):
  """
  Helper function for the mallows distro above.
  For Phi and a given number of candidates, compute the
  insertion probability vectors.

  Method mostly implemented from Lu, Tyler, and Craig Boutilier. 
  "Learning Mallows models with pairwise preferences." 
  Proceedings of the 28th International Conference on Machine Learning (
  ICML-11). 2011.

  Parameters
  -----------
  length: integer
    Number of insertion vectors to compute.  Equal to the number of 
    items that the distribution is over.

  phi: float
    Dispersion parameter.  0.0 gives a unanimous culture while
    1.0 gives the impartial culture.

  Returns
  -----------
  vec_dist: dict
    A mapping from index ---> insertion vector
    where each element of the insertion vector (insert[i]) is the probability
    that candidate[index] is inserted at insert[i].  Hence the the length
    of insert is equal to index+1 (so 1, 2, 3 ....)

  Notes
  -----------
  """
  # For each element in length, compute an insertion
  # probability distribution according to phi.
  vec_dist = {}
  for i in range(length):
    # Start with an empty distro of length i+1
    dist = [0] * (i+1)
    #compute the denom = phi^0 + phi^1 + ... phi^(i-1)
    denom = sum([pow(phi,k) for k in range(len(dist))])
    #Fill each element of the distro with phi^i-j / denom
    for j in range(len(dist)):
      dist[j] = pow(phi, i - j) / denom
      #print(str(dist) + "total: " + str(sum(dist)))
      vec_dist[i] = dist
  return vec_dist

def remove_candidates(orders, candidates_to_remove):
  """
    Remove a set of candidates from a list representing a preference order.
  """
  projection = []
  for c_vote in orders:
    temp_vote = copy.copy(c_vote)
    for c_remove in candidates_to_remove:
      temp_vote.remove(c_remove)
    projection.append(temp_vote)
  return projection

def restrict_score_matrix(score_matrix, assignment):
  """
  Given a score matrix and an assignment, change
  all the elements of the score matrix not present in 
  the assignment to 0.

  Parameters
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  assignment: dict
    A dict from agent --> [agents] which is a list of lenght m
    of agents not in the partition of agent i.

  Returns
  -----------
  restricted_score_matrix: array like
    A copy of the score matrix passed in with every element
    not in the assignment removed.
  
  Notes
  -----------
  This just copies over the valid entries and leaves the rest as zero.
  Be careful when mixing with the normzliation code as it could lead to 
  some strange ass shit...
  
  """

  restricted_score_matrix = np.zeros((len(assignment.keys()), len(assignment.keys())))

  for c_agent, c_assignment in assignment.items():
    for i in c_assignment:
      restricted_score_matrix[i][c_agent] = score_matrix[i][c_agent]

  return restricted_score_matrix

def profile_utilities_to_score_matrix(profile, utilities):
  """
  Given a profile and a COMMON! utility vector, create a score matrix.

  Parameters
  -----------
  profile: dict
    mapping of agents to their orders as a list with list[0] being
    the most prefered.

  utilities: array like
    list of utility mapped to the index of the profile.  this vector must
    be at least as long as the longest order.

  Returns
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  Notes
  -----------
  """
  #Verify that the utility vector is long enough...
  for agent,c_vote in profile.items():
    if len(c_vote) > len(utilities):
      print("Agent vote is longer than utility vector")
      return 0
  
  score_matrix = np.zeros((len(profile.keys()), len(profile.keys())))
  # Otherwise, build a column element...
  for c_agent in profile.keys():
    c_column = np.zeros(len(profile.keys()))
    for i, c_other in enumerate(profile[c_agent]):
      c_column[c_other] = utilities[i]
    score_matrix[:, c_agent] = c_column

  return score_matrix

def profile_classes_to_score_matrix(profile, scores, distribution = []):
  """
  Given a profile and vector of scores, block the candidates
  into size 1/len(score) blocks and assign everyone in that block
  that value.

  Parameters
  -----------
  profile: dict
    mapping of agents to their orders as a list with list[0] being
    the most prefered.

  scores: array like
    score[i] is the score assigned to the fraction 1/len(score) of 
    the agents.  So score[0] is assigned to the first n/len(score)
    agents in my profilea and on.  Will use ceil so it's well defined.

  distribution: array like
    a vector of length scores which defines the width of that 
    score distribution.  Implicitly the score[0] has width
    equal to the first distribution[0] portion of the profile.

  Returns
  -----------
  score_matrix: array like
    The numerical scores of the agents for all the other agents.
    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.

  Notes
  -----------
  The way this is written someone gets every score if possible.  I.e. if score[0]
  and score[1] should go to the top guy only, then the 2nd place guy gets score[1].
  Don't submit stupid score vectors (e.g., scores=[1000, 5000, 0], distribution=[0.0000001, 0.000001, 1-0.0000002])
  as the top and aecond top guy will get 1000 and 5000 respectivly.

  """
  #Verify that the utility vector is long enough...
  if len(scores) <= 1:
    print("Scores vector is <= 1... cannot use.")
    return 0

  score_matrix = np.zeros((len(profile.keys()), len(profile.keys())))
  
  if distribution != []:
    if not np.isclose(sum(distribution), 1.0):
      print("Distribution of scores is not a distribution")
      return 0
    if len(scores) != len(distribution):
      print("Distribution and scores are different lenghts!")
      return 0
  
    if _DEBUG: print("Score: ", str(scores))

    # Otherwise, build a column element...
    for c_agent in profile.keys():
      #Find the right column...
      c_column = np.array([0]*len(profile.keys()))
      for i, c_other in enumerate(profile[c_agent]):
        #Determine the \leq index.
        cv = 0
        p = float(i) / float(len(profile[c_agent])) - distribution[cv]
        while p > 0.0:
          cv+=1
          p -= distribution[cv]
        c_column[c_other] = scores[cv]
      score_matrix[:, c_agent] = c_column

  # Use even distribution...
  else:
    if _DEBUG: print("Score: ", str(scores))
    # Otherwise, build a column element...
    for c_agent in profile.keys():
    # If index(c_agent) <= n/len(score) 
    # points(c_agent) = score[ FLOOR((index(c_agent) / n) / (1./len(score)))  ]
    # Maps percentile to index of score function... hence 1/s% --> 0, 1/(s+1 -- 1/2s --> 1)...
      c_column = np.array([0]*len(profile.keys()))
      for i, c_other in enumerate(profile[c_agent]):
        c_column[c_other] = scores[int(math.floor((float(i)/ float(len(profile[c_agent])))  * float(len(scores))))]  
      score_matrix[:, c_agent] = c_column


  return score_matrix

def strict_m_score_matrix(profile, assignment, scores):
  """
  Given a score matrix and an assignment, change
  all the elements of the score matrix not present in 
  the assignment to 0.  Also score those present
  in the assignment strictly according to the scores
  with score[0] being applied to the most prefered agent.

  Parameters
  -----------
  profile: dict
    mapping of agents to their orders as a list with list[0] being
    the most prefered.

  assignment: dict
    A dict from agent --> [agents] which is a list of lenght m
    of agents not in the partition of agent i.

  scores: array like
    An array of length m which is the points assigned to 
    the most prefered agent to the least prefered agent.

  Returns
  -----------
  restricted_score_matrix: array like
    The numerical scores of the agents for their assigned agents
    according to the scores given.

    We use the convention that a[i,j] is the grade that agent 
    j gives to i.  This means that column j is all the grades
    *given* by agent j and that row i is the grades *recieved*
    by agent i.
  
  Notes
  -----------
  This just copies over the valid entries and leaves the rest as zero.
  Be careful when mixing with the normzliation code as it could lead to 
  some strange ass shit...
  
  """
  #Verify that the utility vector is long enough...
  for k,v in assignment.items():
    if len(v) > len(scores):
      print("Agent assignment is longer than score vector")
      print("Agent " + str(k) + " Assignment: " + str(v) + " Scores: " + str(score))
      exit()
  
  strict_m_matrix = np.zeros((len(profile.keys()), len(profile.keys())))
  # Otherwise, build a column element...
  for c_agent in profile.keys():
    c_column = np.zeros(len(profile.keys()))
    # Sorting tuple is (index, agent)
    p = []
    for i,o in enumerate(profile[c_agent]):
      if o in assignment[c_agent]:
        p.append((i, o))
    # p is now the sorted list of the assignment.
    p = sorted(p, key=lambda x:x[0], reverse=False)
    for i,t in enumerate(p):
      c_column[t[1]] = scores[i]
    strict_m_matrix[:, c_agent] = c_column

  return strict_m_matrix  





  strict_m_matrix = np.zeros((len(assignment.keys()), len(assignment.keys())))

  restricted = restrict_score_matrix(score_matrix, assignment)

  print(restricted)

  return strict_m_matrix
