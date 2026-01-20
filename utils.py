from math import sqrt

# **Review Functions**
def make_review(restaurant_name, score):
    """Creates a review abstraction."""
    return {'restaurant': restaurant_name, 'score': score}

def review_restaurant_name(review):
    """Returns the restaurant name associated with a review."""
    return review['restaurant']

def review_score(review):
    """Returns the score associated with a review."""
    return review['score']

# **Restaurant Data Abstraction**
def make_restaurant(name, location, categories, price, reviews):
    """Return a restaurant data abstraction."""
    return {
        'name': name,
        'location': location,
        'categories': categories,
        'price': price,
        'reviews': reviews  # Ensuring reviews are stored properly
    }

def restaurant_name(restaurant):
    """Return the name of the restaurant, which is a string."""
    return restaurant['name']

def restaurant_location(restaurant):
    """Return the location of the restaurant, which is a list containing latitude and longitude."""
    return restaurant['location']

def restaurant_categories(restaurant):
    """Return the categories of the restaurant, which is a list of strings."""
    return restaurant['categories']

def restaurant_price(restaurant):
    """Return the price of the restaurant, which is a number."""
    return restaurant['price']

def restaurant_scores(restaurant):
    """Return a list of scores based on the reviews of the restaurant."""
    return [review_score(review) for review in restaurant['reviews']]
def map_and_filter(s, map_fn, filter_fn):
    return [map_fn(x) for x in s if filter_fn(x)]

def key_of_min_value(d):
    return min(d, key=lambda k: d[k])

def enumerate(s, start=0):
    return list(map(list, zip(range(start, start + len(s)), s)))
def mean(s):
    return sum(s) / len(s) if s else 0
def distance(loc1, loc2):
    """Return the Euclidean distance between two locations.
    
    Each location is a list of two numbers: [latitude, longitude].
    """
    return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
def zip(*sequences):
    return list(map(list, __import__("builtins").zip(*sequences)))

def enumerate(s, start=0):
    return zip(range(start, start + len(s)), s)
