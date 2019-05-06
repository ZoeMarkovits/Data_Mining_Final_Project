# Data_Mining_Project_Three
## Data Mining - CSC 84040: Project Three 

#### Data Source:
* Data Scraped from BeerAdvocate: https://www.beeradvocate.com/

#### Problems to be Solved:
* Can we create both a content based and a collaborative filtering recommendation system from our given data?

#### Models and Methods Used:
* Content Based Recommender:
  * Text cleaning using NLTK 
  * Topic modeling done with CountVectorizer and NMF
  * Recommendations using cosine similarity 
* Collaborative Filtering Recommender:
  * Matrix factrization done with SVD
  * Recommendations using cosine similarity 
* Web App Built With Flask 
  * https://dm-proj3.herokuapp.com/
