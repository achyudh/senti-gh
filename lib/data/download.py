from requests.auth import HTTPBasicAuth
from flask import Flask
import requests, json, time

with open('config.json', 'r') as config_file:
    client_config = json.load(config_file)

app = Flask(__name__)
http_auth_username = client_config['HTTP_AUTH_USERNAME']
http_auth_secret = client_config['HTTP_AUTH_SECRET']
http_auth = HTTPBasicAuth(http_auth_username, http_auth_secret)

# Header to get reactions along with comments
reactions_header = {'Accept': 'application/vnd.github.squirrel-girl-preview',
                    'direction': 'desc', 'sort': 'created'}

def rate_reset_wait(headers):
    ratelimit_remaining = int(headers['X-RateLimit-Remaining'])
    if ratelimit_remaining <= 0:
        print("Waiting for %d minutes..." % ((int(headers['X-RateLimit-Reset']) - time.time())//60))
        time.sleep(int(headers['X-RateLimit-Reset']) - time.time() + 1)
        return "RateLimit Reset"
    else:
        if ratelimit_remaining % 100 == 0:
            print('X-RateLimit-Remaining:', ratelimit_remaining)
        return "Positive RateLimit"


def generic(request_url, headers=None, num_pages=1):
    merged_response = list()
    for i0 in range(num_pages):
        # print("Request:", request_url)
        if headers is not None:
            response = requests.get(request_url, auth=http_auth, headers=headers)
        else:
            response = requests.get(request_url, auth=http_auth)
        wait_status = rate_reset_wait(response.headers)
        if wait_status == "Positive RateLimit":
            merged_response.extend(response.json())
        else:
            if headers is not None:
                response = requests.get(request_url, auth=http_auth, headers=headers)
            else:
                response = requests.get(request_url, auth=http_auth)
            merged_response.extend(response.json())

        # Change request_url to next url in the link
        if 'Link' in response.headers:
            raw_links = response.headers['Link'].split(',')
            next_url = None
            for link in raw_links:
                split_link = link.split(';')
                if split_link[1][-6:] == '"next"':
                    next_url = split_link[0].strip()[1:-1]
                    break
            if next_url is not None:
                request_url = next_url
            else:
                break
        else:
            break

    return merged_response


def commits(full_repo_name, num_pages=1):
    request_url = 'https://api.github.com/repos/%s/commits' % full_repo_name
    return generic(request_url, headers=reactions_header, num_pages=num_pages)


def pull_requests(full_repo_name, num_pages=1):
    request_url = 'https://api.github.com/repos/%s/pulls' % full_repo_name
    return generic(request_url, headers=reactions_header, num_pages=num_pages)


def issues(full_repo_name, num_pages=1):
    request_url = 'https://api.github.com/repos/%s/issues' % full_repo_name
    return generic(request_url, headers=reactions_header, num_pages=num_pages)


def commit_comments(full_repo_name, num_pages=1, commit_ref=None):
    if commit_ref is None:
        request_url = 'https://api.github.com/repos/%s/comments' % full_repo_name
        return generic(request_url, headers=reactions_header, num_pages=num_pages)
    else:
        request_url = 'https://api.github.com/repos/%s/commits/%s/comments' % full_repo_name, commit_ref
        return generic(request_url, headers=reactions_header)


def review_comments(full_repo_name, num_pages=1, pr_number=None):
    if pr_number is None:
        request_url = 'https://api.github.com/repos/%s/pulls/comments' % full_repo_name
        return generic(request_url, headers=reactions_header, num_pages=num_pages)
    else:
        request_url = 'https://api.github.com/repos/%s/pulls/%s/comments' % full_repo_name, pr_number
        return generic(request_url, headers=reactions_header)


def issue_comments(full_repo_name, num_pages=1, issue_number=None):
    if issue_number is None:
        request_url = 'https://api.github.com/repos/%s/issues/comments' % full_repo_name
        return generic(request_url, headers=reactions_header, num_pages=num_pages)
    else:
        request_url = 'https://api.github.com/repos/%s/issues/%s/comments' % full_repo_name, issue_number
        return generic(request_url, headers=reactions_header)