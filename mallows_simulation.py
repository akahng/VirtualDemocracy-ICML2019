import profile_generator
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
import itertools

def borda(all_rankings):
	# input: list of rankings of alternatives by index
	# output: ordered list of indices (borda)
	score_dict = {}
	first_ranking = all_rankings[0]
	num_rankings = len(all_rankings)
	for x in first_ranking:
		score_dict[x] = first_ranking.index(x)
	for i in range(1, num_rankings):
		curr_ranking = all_rankings[i]
		for x in curr_ranking:
			score_dict[x] += curr_ranking.index(x)
	return score_dict


def count_pairs(profile, alts, p, num_iters):

	num_alts = len(alts)
	pair_dict = {}
	
	for i in range(num_alts):
		for j in range(i+1, num_alts):
			pair_dict[(i,j)] = [0,0]

	for x in range(num_iters):
		noisy_profile = generate_noisy_profile(profile, p)

		noisy_borda_scores = borda(noisy_profile)
		noisy_borda_ranking = [k for k in sorted(noisy_borda_scores, key=noisy_borda_scores.__getitem__)]

		for i in range(num_alts):
			for j in range(i+1, num_alts):
				if noisy_borda_ranking.index(i) < noisy_borda_ranking.index(j): # i better
					pair_dict[(i,j)][0] += 1
				elif noisy_borda_ranking.index(j) < noisy_borda_ranking.index(i): # j better
					pair_dict[(i,j)][1] += 1

	return pair_dict


def generate_noisy_profile(profile_dict, p):
	noisy_profile = []
	for profile in profile_dict.items():
		# profile is something like (0, [1, 4, 2, 0, 3])
		noisy_profile_dict = profile_generator.generate_mallows_mixture_profile([0], profile[1], [1], [profile[1]], [p])
		noisy_profile.append(noisy_profile_dict[0])
	return noisy_profile


def calculate_stdev(a):
	return np.std(np.array(a))

def get_scores_vs_errors(num_voters, num_alts, p, num_iters, mixture):
	# main function: generate profile, count errors, and plot

	voters = list(range(num_voters))
	alts = list(range(num_alts))
	alts_rev = alts[::-1]

	profile = profile_generator.generate_mallows_mixture_profile(voters, alts, [mixture[1], mixture[2]], [alts, alts_rev], [p, p])

	inverse_borda_scores = borda(profile)
	borda_ranking = [k for k in sorted(inverse_borda_scores, key=inverse_borda_scores.__getitem__)]
	score_vs_error = []

	pair_dict = count_pairs(profile, alts, p, num_iters)

	for i in range(num_alts):
		for j in range(i+1, num_alts):
			inverse_borda_diff = inverse_borda_scores[i] - inverse_borda_scores[j]

			if inverse_borda_diff < 0:  # i is better than j
				# count errors: # j > i / total
				[ibetter, jbetter] = pair_dict[(i,j)]
				score_vs_error.append((abs(inverse_borda_diff), jbetter / (ibetter + jbetter)))
			elif inverse_borda_diff > 0:  # j is better than i
				# count errors: # i > j / total
				[ibetter, jbetter] = pair_dict[(i,j)]
				score_vs_error.append((inverse_borda_diff, ibetter / (ibetter + jbetter)))
			else:  # they're indistinguishable
				continue

	# plot scores vs. errors
	plt.clf()
	
	try:
		borda_score_values = [score for score, weight in score_vs_error]
		score_weights = [weight for score, weight in score_vs_error]

		bin_means, bin_edges, binnumber = stats.binned_statistic(borda_score_values, score_weights, 'mean', bins=list(range(0, num_voters * (num_alts - 1), num_voters)))

		bin_stdevs, _, _ = stats.binned_statistic(borda_score_values, score_weights, statistic=calculate_stdev, bins=list(range(0, num_voters * (num_alts - 1), num_voters)))

		x_locations = [x + num_voters / 2 for x in list(range(0, num_voters * (num_alts - 2), num_voters))]
		x_locations = [x / num_voters for x in x_locations]
		bar_width = num_voters / num_voters

		plt.bar(x_locations, bin_means, width = bar_width, yerr=bin_stdevs )
		plt.ylim(bottom=0)
		plt.xlim(left=0, right= num_alts)

		plt.xlabel('Average Borda score difference')
		plt.ylabel('Average error probability')

		plt.savefig(f'{mixture[0]}_{num_voters}voters_{num_alts}alts_{int(p*100)}phi.pdf')
	except:
		print(f'{num_voters}voters_{num_alts}alts_{int(p*100)}phi did not work')
		pass


def main():

	num_voters = 100
	num_alts = 40
	num_iters = 100
	ps = np.arange(0.1,1,0.1)

	mixtures = [['p100',1,0], ['p70',0.7,0.3], ['p50',0.5,0.5]]
	for mixture in mixtures:
		for p in ps:
			get_scores_vs_errors(num_voters, num_alts, p, num_iters, mixture)
			print(f'done with phi={p}, {mixture}')


if __name__ == "__main__":
	main()
