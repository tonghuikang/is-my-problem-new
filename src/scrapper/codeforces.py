from ..utils import read_problems, dump_json_safe
import json
import os
import requests
import time
import bs4
import typing
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import tempfile


scrapped_problems=[]
try:
    scrapped_problems=read_problems('problems/codeforces.json')
    print(f'Recalled {len(scrapped_problems)} scrapped problems')
except:
    print('Cannot find scrapped problems')
scrapped_uids = set(p['uid'] for p in scrapped_problems)

codeforces_endpoint = 'https://codeforces.com/api/problemset.problems'
# get list of problems
list_problems = requests.get(codeforces_endpoint).json()['result']['problems']
print('# problems:',len(list_problems))

# https://stackoverflow.com/a/66835172
def get_text(tag: bs4.Tag) -> str:
    _inline_elements = {"a","span","em","strong","u","i","font","mark","label",
    "s","sub","sup","tt","bdo","button","cite","del","b","a","font",}

    def _get_text(tag: bs4.Tag) -> typing.Generator:
        for child in tag.children:
            if isinstance(child, bs4.Tag):
                # if the tag is a block type tag then yield new lines before after
                is_block_element = child.name not in _inline_elements
                if is_block_element:
                    yield "\n"
                yield from ["\n"] if child.name == "br" else _get_text(child)
                if is_block_element:
                    yield "\n"
            elif isinstance(child, bs4.NavigableString):
                yield child.string
    return "".join(_get_text(tag))

# a scrapper for codeforces
def scrap_problem(contestId, index, rating, tags, uid):
    url = f'https://codeforces.com/contest/{contestId}/problem/{index}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    statement = soup.find(class_='problem-statement')
    try:
        statement.find(class_='header').decompose()
    except:
        pass
    statement_body = statement.find('div')
    statement_body = get_text(statement_body)
    # \r -> \n, remove duplicate \n, strip
    statement_body = statement_body.replace('\r', '\n').replace('\n\n', '\n').replace('$$$','$').strip()
    problem = {
        'uid': uid,
        'url': url,
        'tags': tags,
#        'raw': str(response.content),
        'statement': statement_body,
        'contestId': contestId,
        'index': index,
        'rating': rating,
    }
    return problem

for problem in tqdm(list_problems):
    contestId, index, rating, tags = problem['contestId'], problem['index'], problem['rating'], problem['tags']
    uid = f'Codeforces{contestId}{index}'
    if uid in scrapped_uids:
        continue
    print(f'Scrapping {uid}')
    result = None
    try:
        result = scrap_problem(contestId, index, rating, tags, uid)
    except Exception as e:
        print('Error while scrapping:', e)
    if result is not None:
        scrapped_problems.append(result)
    time.sleep(0.3)
    # save it to file
    dump_json_safe(scrapped_problems, 'problems/codeforces.json')