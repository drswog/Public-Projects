print('Importing libaries and resources')
import time # for profiling
start_time = time.time()
from tabulate import tabulate
import praw
import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import prawcore
import inflect
import string

p = inflect.engine()
from datetime import datetime, timedelta # to convert unix time
today = datetime.today()

import logging
logging.getLogger('praw').setLevel(logging.CRITICAL)

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng') # Download the missing resource
nltk.download('omw-1.4')
from nltk.corpus import stopwords, wordnet as wn
_ = list(wn.all_lemma_names())
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer as lemmatizer

from concurrent.futures import ThreadPoolExecutor, as_completed

print("All libraries and resources loaded!")
end_time = time.time()
print(f"Libaries and resources installation time: {end_time-start_time:.6f} seconds")


def main():
  # Start main program logic
  print("😯Reddit Data Scraper")
  print("by DARIUS :)")
  print('')
  reddit = praw.Reddit(client_id='<insert here>', client_secret='<insert here>', user_agent='<insert here>')

  # Get subreddit(s) names
  subreddits = whatsubreddit_s(reddit)
  print(f"Subreddit(s) to analyse: {', '.join(subreddits)}")

  # How many posts to analyse
  limit = numberofposts()

  # Select metrics to analyse
  metric_s = whatmetric_s() # a list of tuples
  namedmetrics = []
  for tuple in metric_s:
    namedmetric, function = tuple
    namedmetrics.append(namedmetric)
  strofnamedmetrics = ", ".join(namedmetrics)
  print(f"Metrics selected: {strofnamedmetrics}\n")

  # Prompt user for X if most popular topics was selected
  mostpopulartopics_selected = any(func == mostpopulartopics for _, func in metric_s)
  X = None
  if mostpopulartopics_selected:
    while True:
      try:
        X = int(input('Program will show Top X most common words...\nPlease specify X: '))
        if X == 0 or X == '':
          print('X cannot be 0 or an empty string!')
          continue
        else:
          break
      except ValueError:
        pass


  # obtain sort and time_range parameters to insert into metric functions
  sort = obtainsort()
  time_range = obtaintimerange()
  appropriatetimeperiod = timeperiodappropriate(time_range)
  #start_poststoprocess = time.time()
  #subreddits_posts_data = posts_to_process(reddit=reddit, subreddits=subreddits, sort = sort, time_range = time_range, limit=limit, appropriatetimeperiod=appropriatetimeperiod)
  #end_poststoprocess = time.time()
  #print(f'Time taken for subreddits_posts_data: {end_poststoprocess-start_poststoprocess:.2f}s')

  subsdata = {}
  avepessubsdata = {}
  # run threads
  with ThreadPoolExecutor(max_workers=len(subreddits)) as executor: # creates a pool of threads (1 per subreddit)
    futures = []
    for sub in subreddits:
        futures.append(
            executor.submit( # starts process_subreddit() in a separate thread
                process_subreddit,
                reddit, sub, time_range, sort, limit, appropriatetimeperiod, metric_s, subsdata, avepessubsdata, X)
            )
    for future in as_completed(futures): # waits for all threads to finish
        print(future.result())  # prints: Done: r/subreddit

  #print(f"Final mptsubsdata is this: {subsdata}")
  # Show mptdata for one/multiple subreddits, combined into one figure
  for nameofmetric, metricfunc in metric_s:
    if metricfunc == mostpopulartopics:
      showmptdata(X, subsdata, time_range, subreddits, limit, sort)
    if metricfunc == avepes:
      showavepes(avepessubsdata, time_range)
    if metricfunc == showactivity:
      showactivity(reddit, avepessubsdata, subreddits, time_range)

def process_subreddit(reddit,sub,time_range,sort,limit,appropriatetimeperiod, metric_s, subsdata, avepessubsdata, X=None,):
  '''
  - obtains subreddit posts
  IF RESPECTIVE METRIC IS SELECTED...
    - calculate avepes > print thru avepesdata
    - calculate activity
    - calculate mostpopulartopics
  '''
  start_time = time.time()
  x_time_ago = datetimerange(time_range)
  number,timeperiod = time_range
  # initalise onesubdict that stores data for one sub
  subreddits_posts_data = {}
  countpostwithintimerange = 0
  countpostoutsidetimerange = 0
  # for one sub, collate in a list: countpostwithintimerange, countpostoutsidetimerange, postswithintimerange
  if sort in ['top', 'controversial']:
    allposts = list(getattr(reddit.subreddit(sub),sort)(time_filter = appropriatetimeperiod, limit=limit))
    postsintime = [post for post in allposts if x_time_ago <= unixconverter(post.created_utc) <= today]
    postsoutsidetime = [post for post in allposts if not (x_time_ago <= unixconverter(post.created_utc) <= today)]
  elif sort in ['hot','rising','new']:
    allposts = list(getattr(reddit.subreddit(sub),sort)(limit = limit))
    postsintime = [post for post in allposts if x_time_ago <= unixconverter(post.created_utc) <= today]
    postsoutsidetime = [post for post in allposts if not (x_time_ago <= unixconverter(post.created_utc) <= today)]
    countpostoutsidetimerange += len(postsoutsidetime)
    postswithinonemonth = list(getattr(reddit.subreddit(sub),sort)(limit = limit))
  if timeperiod == 'month' and 2 <= int(number) <= 10: # include postswithinonemonth to postswithintimerange -> how does this affect no. of posts in time range and no. of posts outside of time range
    # remove duplicates between postswithinonemonth + subreddit_posts by calling set()
    print("[!]:\n- Detected that you are requesting posts 2-10 months from today.\n- Do note that results may be severely limited, and thus, compensated with posts from the past month.\n- Hence, bar graphs may look similar for such requests.\n")
    if sort in ['top', 'controversial']:
      method = getattr(reddit.subreddit(sub), sort)
      onemonthposts = list(method(time_filter = 'month', limit = limit))
      postswithintimerange = set(onemonthposts + postsintime)
      countpostwithintimerange += len(postswithintimerange)
    elif sort in ['hot','rising','new']:
      postswithintimerange = postsintime
      countpostwithintimerange += len(postswithintimerange)
  else:
    postswithintimerange = postsintime
    countpostwithintimerange += len(postswithintimerange)
  subreddits_posts_data[sub] = [countpostwithintimerange,countpostoutsidetimerange, postswithintimerange]

  # add avepes score to subreddits_posts_data
  avepesdata(subreddits_posts_data, [sub], time_range)

  listofmetrics = []
  for nameofmetric,metricfunc in metric_s:
    listofmetrics.append(metricfunc)

  if (showactivity in listofmetrics) or (avepes in listofmetrics) or (showactivity and avepes in listofmetrics):
     avepes(subreddits_posts_data=subreddits_posts_data, avepessubsdata=avepessubsdata)

  for nameofmetric, metricfunc in metric_s:
    if metricfunc == mostpopulartopics:
      metricfunc(subsdata, subreddits_posts_data=subreddits_posts_data, subreddits=[sub], sort=sort, time_range=time_range, limit=limit, X=X)
  end_time = time.time()
  print(f"process_subreddit execution time: {end_time-start_time:.6f} seconds")
  return f"Posts processed and metric functions executed for r/{sub}"

def posts_to_process(reddit,limit,appropriatetimeperiod,sort,subreddits:list,time_range = (('1','week'))):
  '''
  returns a dictionary with
  {'subreddit name':[countpostwithintimerange,countpostoutsidetimerange, postswithintimerange]}1
  '''
  start_time = time.time()
  x_time_ago = datetimerange(time_range)
  number, timeperiod = time_range
  subredditdict = {}
  for sub in subreddits:
    countpostwithintimerange = 0
    countpostoutsidetimerange = 0
    if sort in ['top','controversial']:
      method = getattr(reddit.subreddit(sub), sort)
      allposts = list(method(time_filter = appropriatetimeperiod, limit = limit))
      postsintime = [post for post in allposts if x_time_ago <= unixconverter(post.created_utc) <= today]
      postsoutsidetime = [post for post in allposts if not (x_time_ago <= unixconverter(post.created_utc) <= today)]
      countpostoutsidetimerange += len(postsoutsidetime)
    # selected sort = hot, rising, new
    elif sort in ['hot','rising','new']:
      method = getattr(reddit.subreddit(sub), sort)
      allposts = list(method(limit = limit))
      postsintime = [post for post in allposts if x_time_ago <= unixconverter(post.created_utc) <= today]
      postsoutsidetime = [post for post in allposts if not (x_time_ago <= unixconverter(post.created_utc) <= today)]
      countpostoutsidetimerange += len(postsoutsidetime)
      postswithinonemonth = list(method(limit = limit))
    # collection of posts within one month
    if timeperiod == 'month' and 2 <= int(number) <= 10: # include postswithinonemonth to postswithintimerange -> how does this affect no. of posts in time range and no. of posts outside of time range
      # remove duplicates between postswithinonemonth + subreddit_posts by calling set()
      print("[!]:\n- Detected that you are requesting posts 2-10 months from today.\n- Do note that results may be severely limited, and thus, compensated with posts from the past month.\n- Hence, bar graphs may look similar for such requests.\n")
      if sort in ['top', 'controversial']:
        method = getattr(reddit.subreddit(sub), sort)
        onemonthposts = list(method(time_filter = 'month', limit = limit))
        postswithintimerange = set(onemonthposts + postsintime)
        countpostwithintimerange += len(postswithintimerange)
      elif sort in ['hot','rising','new']:
        postswithintimerange = postsintime
        countpostwithintimerange += len(postswithintimerange)
    else:
      postswithintimerange = postsintime
      countpostwithintimerange += len(postswithintimerange)
    subredditdict[sub] = [countpostwithintimerange,countpostoutsidetimerange, postswithintimerange]
  end_time = time.time()
  print(f"posts_to_process execution time: {end_time-start_time:.6f} seconds")
  return subredditdict

def showactivity(reddit, avepessubsdata, subreddits:list, time_range = ("1","week")):
  '''
  prints out the activity score for each sub
  - define aveppm as the average no. of posts in the past 24h per member
  - aka total no. of posts in the past 24h / total number of members
  - activity = 0.3(normalised ppm)+0.7(avepesscore)
  '''
  start_time = time.time()
  collateactivitystats = {}
  max_reasonable_ppm = 0.02
  for sub in avepessubsdata:
    print(f'Calculating Activity score for r/{sub}')
    avepesscore = avepessubsdata[sub][2]
    # total posts in past 24h
    total_posts = len([post for post in reddit.subreddit(sub).new(limit=500) if datetimerange(('1','day')) <= unixconverter(post.created_utc) <= today])
    total_subscribers = reddit.subreddit(sub).subscribers
    ppm = total_posts/total_subscribers
    normalized_ppm = math.log(1+ppm)/math.log(1+max_reasonable_ppm)
    activityscore = (0.3*normalized_ppm)+(0.7*avepesscore)
    collateactivitystats[sub] = [total_posts,total_subscribers,activityscore]
  df = pd.DataFrame.from_dict(collateactivitystats, orient='index', columns = ['Total Posts', 'Total Subscribers', 'Activity Score'])
  formatted_df = df.copy()
  formatted_df['Total Subscribers'] = formatted_df['Total Subscribers'].apply(lambda x: f"{int(x):,}")
  formatted_df['Total Posts'] = formatted_df['Total Posts'].apply(lambda x: f"{int(x):,}")
  formatted_df['Activity Score'] = formatted_df['Activity Score'].apply(lambda x: f"{x:.3f}")
  print(tabulate(formatted_df, headers='keys', tablefmt='fancy_grid'))
  print('')
  end_time = time.time()
  print(f"showactivity execution time: {end_time-start_time:.6f} seconds")

def avepesdata(subreddits_posts_data,subreddits:list, time_range = ("1","week"))->float:
  '''
  Appends subreddits_posts_data to include, for each respective sub, the Average Post Engagement Score (avepes) as a float
  from {limit} number of posts sorted by {sort}, in a time range of {time_range}
  '''
  start_time = time.time()
  x_time_ago = datetimerange(time_range)
  number, timeperiod = time_range
  results = []
  for sub in subreddits_posts_data:
    print(f"Calculating Average Post Engagement Score of r/{sub}...")
    postswithintimerange = subreddits_posts_data[sub][2]
    countpostoutsidetimerange = subreddits_posts_data[sub][1]
    countpostwithintimerange = subreddits_posts_data[sub][0]
    # Find the average post engagement score of a post
    alltotalcommentpoints = []
    allpostscores = []
    no_of_posts = 0
    for post in postswithintimerange:
      no_of_posts += 1
      post_totalcommentpoints = totalcommentpoints(post.comments)
      alltotalcommentpoints.append(post_totalcommentpoints)
      allpostscores.append(post.score)
    allpes = []
    max_totalcommentpoints = max(alltotalcommentpoints) if alltotalcommentpoints else 1
    max_postscores = max(allpostscores) if allpostscores else 1

    for i, post in enumerate(postswithintimerange):
      # Ensure division by zero is handled
      if max_totalcommentpoints == 0:
          relativecommentratio = 0
      else:
          relativecommentratio = alltotalcommentpoints[i] / max_totalcommentpoints

      if max_postscores == 0:
          relativescoreratio = 0
      else:
          relativescoreratio = post.score / max_postscores

      pes = (relativecommentratio + relativescoreratio) / 2
      allpes.append(pes)

    if no_of_posts == 0:
      print("No. of posts = 0")
    elif no_of_posts > 0:
      avepesscore = sum(allpes)/no_of_posts
      subreddits_posts_data[sub].append(avepesscore)
  end_time = time.time()
  print(f"avepesdata execution time: {end_time-start_time:.6f} seconds")

def avepes(subreddits_posts_data, avepessubsdata):
  '''
  add [countpostwithintimerange,countpostoutsidetimerange,aveepesscore] to avepessubsdata
  so...
  avepessubsdata = {'terraria':[countpostwithintimerange,countpostoutsidetimerange,aveepesscore],
                    'minecraft': [countpostwithintimerange,countpostoutsidetimerange,aveepesscore]}
  '''
  for sub in subreddits_posts_data:
    totalpostsoutsidetimerange = subreddits_posts_data[sub][1]
    totalpostswithintimerange = subreddits_posts_data[sub][0]
    avepesscore = subreddits_posts_data[sub][3]
    avepessubsdata[sub]=[totalpostsoutsidetimerange,totalpostswithintimerange,avepesscore]
  return avepessubsdata


def showavepes(avepessubsdata, time_range):
  '''
  Prints out a table of the avepes scores for each sub
  '''
  number, timeperiod = time_range
  results = []
  collateavepesstats = {}
  for sub in avepessubsdata:
    countpostoutsidetimerange = avepessubsdata[sub][0]
    countpostwithintimerange = avepessubsdata[sub][1]
    avepesscore = avepessubsdata[sub][2]
    plural_timeperiod = p.plural(timeperiod.capitalize(),number)
    collateavepesstats[sub]= [countpostoutsidetimerange,countpostwithintimerange,avepesscore]
  newdf = pd.DataFrame.from_dict(collateavepesstats, orient='index', columns = ['Total Posts Outside Time Range', 'Total Posts Within Time Range', 'Average Post Engagement Score'])
  print(tabulate(newdf, headers='keys', tablefmt='fancy_grid'))
  print('')

def totalcommentpoints(comments, depth=0, max_depth=3):
  '''
  return the number of points for one post
  '''
  if hasattr(comments, 'replace_more'):
      comments.replace_more(limit=0)
  else:
      return 0 # Return 0 points if comments is not a valid CommentForest
  if depth > max_depth:
    return 0 # limit processing of comments' points to a certain depth level

  points = 0
  # Iterate directly over the CommentForest object
  for comment in comments:
    points += 0.1 * (0.9)**(depth) # points per comment = 1/(depth+1)
    # Recursively call totalcommentpoints on the replies
    points += totalcommentpoints(comment.replies, depth+1, max_depth)
  return points

def timeperiodappropriate(time_range):
  '''
  Returns a str of the appropriate time period
  '''
  number,timeperiod = time_range
  # Determine appropriate time period to insert into time_filter
  n = int(number)
  if timeperiod == 'day':
      if n == 1:
          appropriatetimeperiod = 'day'
      elif 2 <= n <= 7:
          appropriatetimeperiod = 'week'
  elif timeperiod == 'week':
      if n == 1:
          appropriatetimeperiod = 'week'
      elif 2 <= n <= 3:
          appropriatetimeperiod = 'month'
  elif timeperiod == 'month':
      if n == 1:
          appropriatetimeperiod = 'month'
      elif 2 <= n <= 11:
          appropriatetimeperiod = 'year'
  elif timeperiod == 'year':
      appropriatetimeperiod = 'year'
  else:
      appropriatetimeperiod = timeperiod
  return appropriatetimeperiod

def mostpopulartopics(subsdata,subreddits_posts_data, subreddits: list, limit, sort = 'top', time_range = ("1","month"), X = None):
  number,timeperiod = time_range
  # Define date range
  global today
  date_today, time_today = str(today).split(' ')
  x_time_ago = datetimerange(time_range)
  date_x_time_ago, time_x_time_ago = str(x_time_ago).split(' ')
  for sub in subreddits:
    print(f"Finding popular topics for r/{sub}...")
    postswithintimerange = subreddits_posts_data[sub][2]
    countpostoutsidetimerange = subreddits_posts_data[sub][1]
    countpostwithintimerange = subreddits_posts_data[sub][0]

    mostcommonwords = []

    for post in postswithintimerange:
        title = post.title
        description = post.selftext
        title_tokens = processing_nltk(title)
        description_tokens = processing_nltk(description)
        filtered_total_tokens = title_tokens + description_tokens
        words_not_lemmatized, lemmatized_total_tokens = lemmatize_it(filtered_total_tokens)
        final_corpus_of_tokens = words_not_lemmatized + lemmatized_total_tokens

        if final_corpus_of_tokens:
            most_common_word = nltk.FreqDist(final_corpus_of_tokens).max()
            mostcommonwords.append(most_common_word)

    if mostcommonwords:
        freqdist = nltk.FreqDist(mostcommonwords)
        topXwords = freqdist.most_common(X)
        wordx = [word for word, _ in topXwords]
        freqy = [freq for _, freq in topXwords]
        tupledata = (wordx, freqy)

    else:
        print(f"[!] Skipping r/{sub} — no valid text data to extract topics.")

    subsdata[sub] = [countpostoutsidetimerange, countpostwithintimerange, tupledata]
  return subsdata

def showmptdata(X, subsdata, time_range, subreddits, limit, sort):
  global today
  number, timeperiod = time_range
  date_today, time_today = str(today).split(' ')
  x_time_ago = datetimerange(time_range)
  date_x_time_ago, time_x_time_ago = str(x_time_ago).split(' ')

  for sub in subsdata.keys():
    countpostoutsidetimerange = subsdata[sub][0]
    countpostwithintimerange = subsdata[sub][1]
  plural_timeperiod = p.plural(timeperiod.capitalize(),number)

  num_subreddits = len(subreddits)
  # Base dimensions + extra width per subreddit
  fig_width = max(8, num_subreddits * 6)  # Minimum 8", +6" per subreddit
  fig_height = max(6, X * 0.7)  # 0.7" per bar (adjust if labels are long)

  fig, axs = plt.subplots(
      1, num_subreddits,
      figsize=(fig_width, fig_height),
      squeeze=False  # Ensures axs is always a 2D array
  )
  axs = axs.flatten()  # Simplify indexing
# Configure spacing between subplots
  plt.subplots_adjust(
      left=0.1,      # Space for y-axis labels
      right=0.95,    # Prevents right-edge clipping
      top=0.85,      # Space for suptitle
      bottom=0.15,   # Space for x-axis labels
      wspace=0.4     # Horizontal gap between subplots
  )

  for idx, sub in enumerate(subsdata.keys()):
    ax = axs[idx] if num_subreddits > 1 else axs[0]  # Handle single-subreddit case
    xdata, ydata = subsdata[sub][2]

    # Plot data
    bars = ax.barh(xdata, ydata)
    ax.set_title(f'r/{sub}')
    ax.invert_yaxis()  # Highest frequency at top

  plt.suptitle(f'Top {X} most common words among {limit} {sort} posts in {number} {plural_timeperiod} (From {date_x_time_ago} to {date_today})')
  # Save with tight bounding box
  filename = f"top_{X}_words.png"
  plt.savefig(
      filename,
      dpi=300,
      bbox_inches="tight",  # Critical for preventing crop
      pad_inches=1.0        # Extra padding around the figure
  )
  print(f"Plot saved as '{filename}'")
  # Attempt inline display (works in Colab/Jupyter)
  try:
      from IPython.display import Image, display
      display(Image(filename))
  except:
      print("Rendered plot saved to 'project' folder. Open it manually.")
  finally:
      plt.close()  # Free memory

def processing_nltk(text):
  new_tagged_tokens = []
  # tokenize and tag input text
  tagged_tokens = nltk.pos_tag(word_tokenize(text))
  # remove contractions, prepositions, adverbs, conjunctions
  for token in tagged_tokens:
    word, pos_tag = token
    if pos_tag not in ['PRP', 'VBP', 'VBZ', 'MD', 'RB', 'IN', 'CC', 'RBR', 'RBS']:
      new_tagged_tokens.append(word)
    else:
      pass
  # remove tokens that match individual punctuations
  extended_string_punctuation = string.punctuation + '’' + '“' + '”' + '`' + '‘'
  final_tagged_tokens = [token.lower() for token in new_tagged_tokens if token not in extended_string_punctuation]
  # clean off remaining punctuations
  cleaned_tokens = []
  for token in final_tagged_tokens:
    cleaned_token = "".join(char for char in token if char not in extended_string_punctuation)
    # if cleaned_token is empty str
    if cleaned_token == '':
      pass
    elif cleaned_token != '':
      cleaned_tokens.append(cleaned_token)
  # check for negative numbers
  final_tokens = []
  for t in cleaned_tokens:
    if re.search(r"-\d*\.?\d+",t):
      pass
    else:
      final_tokens.append(t)
  # slang_contractions
  slang_contractions = [
    # Online slang contractions
    "brb", "lol", "lmao", "rofl", "omg", "idk", "imo", "imho", "btw", "ttyl", "np",
    "thx", "ty", "yw", "smh", "tbh", "irl", "fyi", "bff", "nvm", "afaik", "idc", "ikr",
    "wtf", "wth", "ily", "omw", "ftw", "fml", "gg", "gr8", "b4", "cya", "plz", "u",
    "ur", "r", "dm", "jk", "tmi", "hmu", "lmk", "atm", "bc", "g2g", "wyd", "wbu",
    "yolo", "rn", "cu", "xoxo", "bf", "gf",

    # Informal English contractions without apostrophes
    "im",     # I'm
    "ive",    # I've
    "id",     # I'd
    "ill",    # I'll
    "hes",    # he's
    "shes",   # she's
    "its",    # it's
    "wont",   # won't
    "cant",   # can't
    "dont",   # don't
    "doesnt", # doesn't
    "didnt",  # didn't
    "isnt",   # isn't
    "arent",  # aren't
    "wasnt",  # wasn't
    "werent", # weren't
    "wouldnt",# wouldn't
    "couldnt",# couldn't
    "shouldnt",# shouldn't
    "aint",   # ain't
    "yall",   # you all
    "gonna",  # going to
    "wanna",  # want to
    "gotta",  # got to
    "lemme",  # let me
    "gimme",  # give me
    "kinda",  # kind of
    "sorta",  # sort of
    "outta",  # out of
    "lotta",  # lot of
    "coulda", # could have
    "shoulda",# should have
    "woulda", # would have
    "mighta", # might have
    "musta",  # must have
    "dunno",  # don't know
    "cmon",   # come on
    "ya",     # you
    "em",     # them
    "dat",    # that (AAVE/slang)
    "dis",    # this (AAVE/slang)
    "dem",    # them (AAVE/slang)
    "got",
    "bro"
]
  stop_words = set(stopwords.words('english'))
  # remove tokens that are stopwords, remove tokens that are numeric
  frfinaltokens = [w for w in final_tokens if not w.lower() in stop_words and not w.lower() in slang_contractions and not w.isnumeric()]
  return frfinaltokens

def lemmatize_it(tokens):
# find the root word of each word in title an description aka stemming
  unlemmatized_tokens = []
  lemmatized_tokens = []
  lemmazer = lemmatizer()
  for w in tokens:
    if lemmazer.lemmatize(w.lower()) == w.lower():
      unlemmatized_tokens.append(w.lower())
    else:
      lemmatized_tokens.append(lemmazer.lemmatize(w.lower()))
  return unlemmatized_tokens, lemmatized_tokens

def obtainsort() -> str:
  '''
  Returns a str of the sort to obtain posts from
  - top/controversial/hot/rising/new
  '''
  print("Select sort by entering corresponding number\n[1] top\n[2] controversial\n[3] hot\n[4] rising\n[5] new")
  while True:
    sort_input = input("Select sort: ")
    sortdict = {"1": 'top',
                "2": 'controversial',
                "3": 'hot',
                "4": 'rising',
                "5": 'new'}
    if sort_input == '':
      print("You must enter a number!")
      continue
    if sort_input != '' and sort_input not in sortdict:
      print("Please enter a number corresponding to a sort")
      continue
    if sort_input in sortdict: # if user inputted smt for sort_input and number selected exists as a key in sortdict
        sort = sortdict[sort_input]
    print(f"Sort selected: {sort}")
    print('')
    return sort


def obtaintimerange() -> tuple:
  '''
  Returns time_range in the format (x,y) where x = number of time period and y = time period
  - Time period: day/week/month/year
  - respective number range for each time period: 1-30/1-3/1-11/1
  '''
  print("Enter time range, in the following format of x,y\n- x is the number of the specified time period, y(day/week/month/year)\n- NOTE!: Value of y ranges for each time period, as seen below👇\nday: 1-7\nweek: 1-3\nmonth: 1-11\nyear: 1")
  timerangedict = {'day':range(1,8),
                     'week':range(1,4),
                     'month':range(1,12),
                     'year':range(1,2)}
  while True:
    time_input = input("Input time range in the above format x,y: ")
    if time_input: # if user inputted smt for time_input
      # no commas
      if time_input.count(',') == 0 or time_input.count(',') > 1:
        print("Please input time range in format of x,y")
        continue
      # one comma
      if time_input.count(',') == 1:
        xylist = time_input.lower().split(',')
        time_range = (xylist[0],xylist[1])
        number,timeperiod = time_range
        # inputted timeperiod is valid
        if timeperiod in timerangedict:
          # inputted number is invalid
          if int(number) not in timerangedict[timeperiod]:
            print("Enter valid number range for specific time period")
            continue
          else:
            print(f"Time range selected: {time_range[0]} {time_range[1]}")
            print('')
            return time_range
        # inputted timeperiod is invalid
        else:
          print('Enter valid time period')
          continue

def whatmetric_s():
    """
    Returns metric(s) selected by the user.
    If one metric selected, returns a single tuple.
    If multiple metrics selected, returns a list of tuples.
    """
    dictmetrics = {
        "1": ("Most Popular Topics", mostpopulartopics),
        "2": ("Activity", showactivity),
        "3": ("Average Post Engagement", avepes)
    }

    print("\nSelect metric(s) to analyse")
    print(">> Enter the corresponding number(s), separated by commas")
    print("   Example: 1 or 1,2")
    print("[1] Most popular topics\n[2] Activity\n[3] Average Post Engagement")

    while True:
        metricslist = input("What metric(s) to analyse?: ").replace(" ", "")
        numbers = metricslist.split(',')

        # Remove empty strings from accidental trailing commas
        numbers = [num for num in numbers if num]

        if not numbers:
            print("No metrics entered. Please try again.\n")
            continue

        # Check if all entries are numeric
        if not all(num.isnumeric() for num in numbers):
            print("To select metric(s), please enter only number(s) corresponding to the options.\n")
            continue

        # Check if all entries are within allowed range
        if any(num not in dictmetrics for num in numbers):
            print("Number you entered is not 1, 2, or 3. Please select appropriate number(s).\n")
            continue

        # Check for duplicates
        if len(set(numbers)) != len(numbers):
            print("Please input distinct numbers only.\n")
            continue

        # Now check number of selections
        if len(numbers) == 1: # one number
            return [dictmetrics[numbers[0]]]
        elif 1 < len(numbers) <= 3: # 2 or 3 numbers
            return [dictmetrics[num] for num in numbers]
        else:
            print("Select up to 3 metrics only.\n")

def whatsubreddit_s(reddit):
  '''
  returns list of subreddit(s) to analyse or a string of subreddit to analyse
  '''
  while True:
      try:
        subredditslist = input("What subreddit(s) to analyse?: ") # nationalservicesg,sgexams,politics
        if subredditslist == '':
          print("Please enter at least one subreddit to analyse")
          continue
        else:
          if subredditslist.count(',') == 0:
            if subreddit_exists(subreddit=subredditslist, reddit=reddit):
              subreddit = [subredditslist.lower()]
              return subreddit
            else:
              print("Please input subreddit(s) again. The subreddit you have entered does not exist.")

          elif subredditslist.count(',') > 0:
            substocheck = subredditslist.lower().split(',')
            subreddits = []
            nonexistentsubs = []
            for sub in substocheck:
              if subreddit_exists(subreddit=sub, reddit=reddit):
                subreddits.append(sub)
                pass
              else:
                nonexistentsubs.append(sub)
            if len(nonexistentsubs) == 0:
              setsub = set(subreddits)
              finalsubreddits = [sub for sub in setsub]
              return finalsubreddits
            elif len(nonexistentsubs) > 0:
              print("Please input subreddit(s) again. The following subreddits do not exist: ")
              for sub in nonexistentsubs:
                print(sub)
      except ValueError:
        pass

def subreddit_exists(subreddit, reddit):
  '''
  returns True if subreddit exists, otherwise returns False
  '''
  try:
      subreddit = reddit.subreddit(subreddit)
      subreddit.id  # Forces a fetch
      return True
  except (prawcore.exceptions.Redirect,prawcore.exceptions.NotFound) :
      # Happens when subreddit does not exist or Subreddit exists but inaccessible
      return False
  except prawcore.exceptions.Forbidden:
      # Subreddit exists but is private or banned
      return True
  except prawcore.exceptions.BadRequest:
      # Invalid subreddit name, e.g '1'
      return False

def numberofposts():
  '''
  Returns a number, referring to the number of posts to analyse aka the limit
  '''
  while True:
    number = input("How many posts to analyse for each subreddit?: ")
    if number.isnumeric():
      return int(number)
    elif not number.isnumeric():
      print("Enter only numbers!")
      continue

def datetimerange(time_range: tuple):
    '''
    returns datetime.datetime object
    '''
    number, timeperiod = time_range
    global today
    timedeltadict = {'day':int(number),
                    'week':7*int(number),
                    'month':31*int(number),
                    'year': 365}
    return today-timedelta(days = timedeltadict[timeperiod])

def unixconverter(timestamp):
    """Converts UTC timestamp to normal DateTime format"""
    return datetime.fromtimestamp(timestamp)

if __name__ == "__main__":
    main()
