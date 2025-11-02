# Reddit Data Scraper
## Video demo: https://youtu.be/7KMl33QE2EM
## Overview
The overaching purpose of this program is to enable users to investigate subreddit engagement metrics over customizable time ranges.

Heavily utilises the *PRAW API* to obtain the subreddit data, *NLTK* to process text data extracted from subreddit user submissions and comments, and *concurrent.futures* to run threads and process multiple subreddits simultaneously.

Secondary libaries such as *pandas* and *matplotlib* are used in data table structuring (using *tabulation*) and graph figure plots.

Key features include:
- Analysis of the **Most Popular Topics** via NLP techniques on post titles and bodies.
- Calculation of the **Average Post Engagement Score (avepes)**, using post scores (upvotes/downvotes) and comment depth-weighted points.
- Computation of a combined **Activity Score**, reflecting both subreddit member participation and engagement.
---
#### User requirements!!!
- Before the program can be executed, the user must install the specified libaries in requirements.txt, as well as provide client_id, client_secret and user_agent in *main()* for PRAW API. [(More information on how to create and retrieve these details found here)](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps)

---
## Main Program Logic
- Imports and loads required libraries and NLTK corpora.
- Prompts the user to enter their subreddit(s), metrics of interest, sorting method, and time window.
- Uses multithreading to process each subreddit independently.
- Calls and manages output of different metric functions, including plotting results where appropriate.

### `process_subreddit()` and `posts_to_process()`
These functions perform the **core scraping** of posts within a specific subreddit. They:
- Fetch posts using selected sort criteria and time filter.
- Categorize posts by whether they fall within or outside the time range.
- Optionally include extra recent posts from the past month for 'top' and 'controversial' sorts when time windows are sparse (2-10 months) to compensate.
	- Ensured such posts are distinct from the remaining posts from the original fetch request.
	- Reddit’s API doesn’t support exact time slicing
	- Third party solutions to facilitate detailed time slicing (e.g pushshift.io) currently unavailable

### `avepesdata()` and `avepes()`
Average Post Engagment Score of a Subreddit, in a given time frame, measures the average level of user engagement for a post.

These functions compute the **Average Post Engagement Score (avepes)**. This score is derived from two factors:
- Normalized comment "points" based on depth in the thread.
	- Using totalcommentpoints(), comment points are obtained by recursively going down the comment tree for one post, setting a max_depth= 3.
- Normalized upvote score.
They then append this result to dictionaries for cross-subreddit comparison.
	- Post Engagement Score = (relativecommentratio + relativescoreratio)/2
		- relativecommentratio = totalcommentpoints of post / highest total comment points in subreddit
		- relativescoreratio = score of one post / highest score of a post in subreddit
		- sum of both ratios is divided by two since each ratio is weighed out of one.

### `showactivity()`
Generates and displays the **Activity Score** for each subreddit.
Activity is defined as a measure of a subreddit's activeness.
This combines a post-per-member metric (PPM) and the avepes score, weighted at 30% and 70% respectively, to offer a more robust indicator of community activeness.
- Activity Score = 0.3(normalised PPM) + 0.7(average pes score)
	- Reasonable PPM metric set at 0.02 (2 posts per 100 members), a value of compromise to prevent any extreme outlier from skewing results.
	- Actual PPM of subreddit = total no. of posts / total subscribers
	- Normalised PPM = lg(1+actual PPM)/lg(1+0.02)

### `mostpopulartopics()` and `showmptdata()`
These are responsible for extracting and displaying the **top X most frequent topic words** from post titles and bodies. NLP cleaning steps (tokenization, stopword filtering, lemmatization) are applied using NLTK. The final data is plotted using horizontal bar charts via `matplotlib`.

### NLP Utilities
Functions such as `processing_nltk()` and `lemmatize_it()` provide **custom text preprocessing** by:
- Removing informal contractions and slang.
- Filtering by Part-Of-Speech (POS) tags and punctuation. (remove contractions, prepositions, adverbs, conjunctions)
- Lemmatizing non-root words to ensure more accurate frequency distribution.

### Miscellaneous Helper Functions
- `whatmetric_s()`, `whatsubreddit_s()`, `obtaintimerange()`, and `obtainsort()` are **user interface functions** that ensure the user selects valid and distinct configurations.
- `unixconverter()` and `datetimerange()` are **date/time helpers** for parsing and validating time ranges.
- `subreddit_exists()` ensures user-provided subreddit names are valid via PRAW API error checking.

---

## Design Choices & Reasoning

- **ThreadPoolExecutor for Concurrency**
   Multithreading was chosen to process multiple subreddits simultaneously, significantly improving performance on larger subreddit sets. Each subreddit is independently queried, processed, and stored.

- **Normalized Scoring (avepes and activity)**
   Instead of relying on raw counts or scores (which vary drastically across subreddit sizes), the metrics normalize inputs to provide **relative measures of engagement**. This makes comparisons across different subreddit scales meaningful.

-  **Natural Language Processing Pipeline**
   Instead of simply counting word frequencies, the NLP pipeline includes:
   - Tokenization and POS filtering (to focus on nouns, adjectives, etc.).
   - Stopword and slang filtering.
   - Lemmatization and cleaning.
   These decisions improve topic word extraction and avoid noise from low-signal tokens.

- **Flexible Time Ranges and Sorts**
   Users can select from Reddit’s available sort types (`top`, `hot`, etc.) and define granular time frames, like “2 weeks” or “3 months.”

---

## Final Thoughts
As a fellow newbie in python and coding in general, in figuring out how to start extracting the general topic from a post, I consulted ChatGPT (unfortunately :/ ) on how to extract popular topics from a sea of posts. I was presented with suggestions using either 'Basic Word Frequency' or 'Latent Dirichlet Allocation (LDA)' Topic Extraction. LDA is a topic modelling method in natural language processing (NLP).

While LDA Topic Extraction is more apt for my use case, I wanted to experience what it was like to manually construct code that extracts, cleans and tokenises from a corpora. I did not want to only install libaries and let the library functions and classes handle and automate away all of my logic like a 'black box', without personally experiencing how to derive the logic and build to code execute such logic. Thus, I chose Basic Word Frequency as the basis for how I would derive the most popular topics, since it would also be something easier for me to start off with, without getting too deep into the knitty gritty of natural language processing.

Further improvements to include:
- Caching results for repeated queries.
- Web-based UI
- Support for CSV/JSON exports and deeper NLP (e.g., topic modeling methods like LDA).
- Add a way to return to previous prompt and re-select
- Especially with r/nationalservicesg and its frequently used acronyms like 'ns' being lemmatised to 'n', enhance the lemmatisation process such that program identifies acronyms to be exempted from lemmatisation.
- Consideration of post flairs in topic extraction
- Detection of moderator bot comments, to be excluded from usage in totalcommentopoints

---

## Author
Hi, I am Darius, and this is my final project for CS50P. It is embarassing to admit how much time I have spent on this project, but I am quite happy that it is completed to a standard where its base utility is to my satisfaction.

For some context, I am a frequent user of Reddit so I have always been fascinated in being able to play or toy with data collected from certain subreddits, and what insights I can ascertain from that data. So with this opportunity to present a personal project for submission to CS50P, my choices for what to pursue for the project had, already, been personally well defined.
