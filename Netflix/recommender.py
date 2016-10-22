from numpy import *
from scipy import optimize

class Recommender:
	def __init__(self):
		# define the number of movies in our 'database'
		"""

		:rtype :
		"""
		self.num_movies = 10
		# define the number of users in our 'database'
		self.num_users = 5
		self.usernames = ['pranay', 'parag','paras', 'parth', 'rahul', 'rahulR']
		self.movies = ['The Wizard of Oz', 'All About Eve ', 'The Godfather', 'King Kong', 'The Adventures of Robin Hood',
		'The Dark Knight', 'Avengers', 'X-Men', 'Toy Story', 'Zootopia']

	def trainig(self):
		self.ratings = random.randint(11, size = (self.num_movies, self.num_users))
		self.did_rate = (self.ratings != 0) * 1
		user_ratings = zeros((self.num_movies, 1))
		# I rate 3 movies
		user_ratings[0] = 8
		user_ratings[4] = 7
		user_ratings[7] = 3
		# Update ratings and did_rate
		self.ratings = append(user_ratings, self.ratings, axis = 1)
		self.did_rate = append(((user_ratings != 0) * 1), self.did_rate, axis = 1)
		self.ratings, self.ratings_mean = self.normalize_ratings(self.ratings, self.did_rate)
		self.num_users = self.ratings.shape[1]
		self.num_features = 3
		# Initialize Parameters theta (user_prefs), X (movie_features)
		self.movie_features = random.randn( self.num_movies, self.num_features )
		self.user_prefs = random.randn( self.num_users, self.num_features )
		self.initial_X_and_theta = r_[self.movie_features.T.flatten(), self.user_prefs.T.flatten()]
		reg_param = 30
		# perform gradient descent, find the minimum cost (sum of squared errors) and optimal values of X (movie_features) and Theta (user_prefs)
		minimized_cost_and_optimal_params = optimize.fmin_cg(self.calculate_cost, fprime=self.calculate_gradient, x0=self.initial_X_and_theta,
			args=(self.ratings, self.did_rate, self.num_users, self.num_movies, self.num_features, reg_param), maxiter=100, disp=True, full_output=True )
		cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]
		self.movie_features, self.user_prefs = self.unroll_params(optimal_movie_features_and_user_prefs, self.num_users,
															 self.num_movies, self.num_features)


	def normalize_ratings(self, ratings, did_rate):
		self.num_movies = ratings.shape[0]
		self.ratings_mean = zeros(shape = (self.num_movies, 1))
		self.ratings_norm = zeros(shape = ratings.shape)
		for i in range(self.num_movies):
			# Get all the indexes where there is a 1
			idx = where(self.did_rate[i] == 1)[0]
			self.ratings_mean[i] = mean(self.ratings[i, idx])
			self.ratings_norm[i, idx] = self.ratings[i, idx] - self.ratings_mean[i]

		return self.ratings_norm, self.ratings_mean


	def unroll_params(self, X_and_theta, num_users, num_movies, num_features):
		# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
		# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
		first_30 = X_and_theta[:num_movies * num_features]
		# Reshape this column vector into a 10 X 3 matrix
		X = first_30.reshape((num_features, num_movies)).transpose()
		# Get the rest of the 18 the numbers, after the first 30
		last_18 = X_and_theta[num_movies * num_features:]
		# Reshape this column vector into a 6 X 3 matrix
		theta = last_18.reshape(num_features, num_users ).transpose()
		return X, theta

	def calculate_gradient(self, X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
		X, theta = self.unroll_params(X_and_theta, num_users, num_movies, num_features)

		# we multiply by did_rate because we only want to consider observations for which a rating was given
		difference = X.dot( theta.T ) * did_rate - ratings
		X_grad = difference.dot( theta ) + reg_param * X
		theta_grad = difference.T.dot( X ) + reg_param * theta

		# wrap the gradients back into a column vector
		return r_[X_grad.T.flatten(), theta_grad.T.flatten()]


	def calculate_cost(self, X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
		X, theta = self.unroll_params(X_and_theta, num_users, num_movies, num_features)
		# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
		cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
		# '**' means an element-wise power
		regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
		return cost + regularization

	def recommend(self, username):
		# Make some predictions (movie recommendations). Dot product
		all_predictions = self.movie_features.dot( self.user_prefs.T )
		# add back the ratings_mean column vector to my (our) predictions
		index = self.usernames.index(username)
		predictions_for_user = all_predictions[:, index:1+index] + self.ratings_mean
		indeices = predictions_for_user.argsort(axis=0)[::-1]
		predictions_for_user =  predictions_for_user[indeices]
		i=0
		j=0
		while i<range(10):
			# print i
			if self.did_rate[indeices[i][0]][index] == 0:
				print self.movies[i] + ': ' + str(predictions_for_user[i][0])
				j+=1
				if j == 3:
					break
			i+=1

if __name__ == '__main__':
	reco = Recommender()
	reco.trainig()
	# print reco.did_rate
	reco.recommend('pranay')