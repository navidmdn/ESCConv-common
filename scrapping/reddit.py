from scrapping.credentials import client_secret, client_id, user_agent, username, password

import praw
import json

# Create a Reddit instance
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    username=username,
    password=password
)

# List of subreddits you want to scrape
# subreddits = [
#     'mentalhealthsupport', 'offmychest', 'sad', 'suicidewatch',
#     'anxietyhelp', 'depression', 'depressed', 'depression_help'
# ]

subreddits = ['mentalhealthsupport']

# Number of posts you want to scrape
post_limit = 2

# Define a function to recursively get comments with more than 10 upvotes
def get_comments(comments):
    comments_list = []
    for comment in comments:
        if comment.score > 10 and not comment.stickied:
            comments_data = {
                'id': comment.id,
                'author': str(comment.author),
                'score': comment.score,
                'body': comment.body
            }
            # Check for replies to the comment
            if len(comment.replies) > 0:
                comments_data['replies'] = get_comments(comment.replies)
            comments_list.append(comments_data)
    return comments_list

# Store all the conversations in a list
conversations = []

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    print(f"Scraping {post_limit} posts from r/{subreddit_name}")

    for post in subreddit.top(limit=post_limit):
        post_data = {
            'title': post.title,
            'id': post.id,
            'url': post.url,
            'score': post.score,
            'selftext': post.selftext,
            'comments': []
        }

        # Fetching the comments of the post
        post.comments.replace_more(limit=None)  # This line ensures you get all comments, not just the top-level
        post_data['comments'] = get_comments(post.comments.list())

        # Append the post data to the conversations list
        conversations.append(post_data)

# Save the conversations to a JSON file
with open('reddit_conversations.json', 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=4)

print("Scraping completed and data stored in 'reddit_conversations.json'")