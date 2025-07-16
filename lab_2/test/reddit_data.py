import os
import praw
from dotenv import load_dotenv

load_dotenv()


class RedditData:
    def __init__(self, subreddit_name, limit=100):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        self.subreddit_name = subreddit_name
        self.limit = limit

    def get_posts(self):
        subreddit = self.reddit.subreddit(self.subreddit_name)
        posts = []
        for post in subreddit.hot(limit=self.limit):
            posts.append(post.title + " " + post.selftext)
        return posts
