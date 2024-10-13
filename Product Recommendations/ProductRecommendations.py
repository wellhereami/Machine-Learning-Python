import pandas
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


folderpath = "movies/"
moviespath = folderpath + "movies.csv"
ratingspath = folderpath + "ratings.csv"

movieData = pandas.read_csv(moviespath)
movieFeatures = ['movieId', 'title']
movieData = movieData[movieFeatures]

ratingsData = pandas.read_csv(ratingspath)
ratingsFeatures = ['userId', 'movieId', 'rating']
ratingsData = ratingsData[ratingsFeatures]

movieThreshold = 50
userThreshold = 50

movieCount = pandas.DataFrame(ratingsData.groupby('movieId').size(), columns=['count'])
popularMovies = movieCount.query('count >= @movieThreshold')
popularMovies = list(set(popularMovies.index))
moviesFilter = ratingsData.movieId.isin(popularMovies).values

usersCount = pandas.DataFrame(ratingsData.groupby('userId').size(), columns=['count'])
activeUsers = usersCount.query('count >= @userThreshold')
activeUsers = list(set(activeUsers.index))
usersFilter = ratingsData.userId.isin(activeUsers).values

ratingsDataframe = ratingsData[moviesFilter & usersFilter]

movieFeatures = ratingsDataframe.pivot(index='movieId', columns='userId', values='rating')
movieFeatures = movieFeatures.fillna(0)

movieMatrix = csr_matrix(movieFeatures.values)
model = NearestNeighbors(metric='cosine')
model.fit(movieMatrix)

movieNameToIndex = {}
index = 0
for id in movieFeatures.index:
    movieName = movieData.loc[movieData['movieId'] == id, 'title'].values[0]
    movieName = movieName.split(" (")[0].lower()
    movieNameToIndex[movieName] = index
    index += 1

answer = input("Enter a movie title: ")
answer = answer.lower()
    
index = movieNameToIndex.get(answer)

if index != None:

    distances, indices = model.kneighbors(movieMatrix[index], n_neighbors=6)

    recommendIndices = indices.flatten()
    for i in range(0, len(recommendIndices)):
        index = recommendIndices[i]
        for name, id in movieNameToIndex.items():
            if(index == id):
                movieName = name
                break
        if i == 0:
            print("Recommendations for " + movieName + ": \n")
        else:
            print(str(i) + ": " + movieName)
else:
    print(answer + " was not found!")