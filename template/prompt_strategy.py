io_prompt = """# Input:
## User's History News:
${history}
## Candidate News:
${candidate}
# Output Format:
Rank candidate news based on the user's history news in the format: "Ranked news: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>"."""

io_pure_prompt = """# Input:
## User's History News:
${history}
## Candidate News:
${candidate}
# Output Format:
Rank candidate news based on the user's history news with the given format."""

io_json_prompt = """# Input:
## User's History News:
${history}
## Candidate News:
${candidate}
# Output Format:
Rank candidate news based on the user's history news and response in the following JSON format:
```json
{
    "ranking": ["C#", "C#", "C#", "C#", "C#", "C#", "C#", "C#", "C#", "C#"]
}
```
YOU SHOULD USE INDEX NUMBER LIKE "C1" INSTEAD OF HEADLINES AND OUTPUT THE RANKING LIST ONLY."""

cot_prompt = """Based on the user's news history, think step by step and recommend candidate news articles.
# Input
## User's History News
${history}
## Candidate News
${candidate}
# Output Format
Rank candidate news based on the user's history news in the format: "Ranked news: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>"."""

cot_first_prompt = """Based on the user's news history, think step by step and recommend candidate news articles.
# Input
## User's History News
${history}
## Candidate News
${candidate}
# Output Format
Rank candidate news based on the user's history news and first output the ranked list in the format: "Ranked news: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>"."""

cot_pure_prompt = """Based on the user's news history, think step by step and recommend candidate news articles.
# Input
## User's History News
${history}
## Candidate News
${candidate}
# Output Format
Rank candidate news based on the user's history news with the given format."""

cot_rankonly_prompt = """Based on the user's news history, think step by step and recommend candidate news articles.
# Input
## User's History News
${history}
## Candidate News
${candidate}
# Output Format
Rank candidate news based on the user's history news in the format: "Ranked news: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>". YOU SHOULD USE INDEX NUMBER LIKE "C1" INSTEAD OF HEADLINES AND OUTPUT THE RANKING LIST ONLY."""

cot_json_prompt = """Based on the user's news history, think step by step and recommend candidate news articles.
# Input
## User's History News
${history}
## Candidate News
${candidate}
# Output Format
Your response should be in the following JSON format:
```json
{
    "ranking": ["C#", "C#", "C#", "C#", "C#", "C#", "C#", "C#", "C#", "C#"]
}
```
YOU SHOULD USE INDEX NUMBER LIKE "C1" INSTEAD OF HEADLINES AND OUTPUT THE RANKING LIST ONLY."""

cot_rec_prompt = """Based on the user's news history, analyze and recommend candidate news articles that align with the user's interests. Use a step-by-step approach to ensure the recommendations are semantically related to the user's preferences.

# Task Description
The goal is to recommend news articles from the 'Candidate News' that the user is most likely to be interested in, based on their 'User's History News'. The recommendations should be made solely on the basis of the user's interests, without considering the order of appearance in the 'Candidate News'. The task is to select news that is most relevant to the user's demonstrated interests.

# Recommendation Process
1. Analyze 'User's History News' to summarize the user's interests. Extract keywords and group them by meaning to form "Topics" that represent related concepts.
2. Extract keywords from 'Candidate News' and evaluate how well they match the user's topics of interest.
3. Ensure that the recommendation is not influenced by the position of the news in 'Candidate News', but solely by the semantic relevance between the candidate news and the user's interests.

# Input
## User's History News
${history}
## Candidate News
${candidate}

# Output Format
Summarize the user's interests and rank the candidate news according to their relevance to these interests. Provide a ranked list in the format: "Ranked news: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>". Additionally, include a summary of the user's interests and an explanation for the ranking of the recommendations.
Output the ranked list first and then explain the results.
"""

rec_cot_prompt = """Based on the user's news history, analyze and recommend candidate news articles that align with the user's interests. Use a step-by-step approach to ensure the recommendations are semantically related to the user's preferences.

# Task Description
The goal is to recommend news articles from the 'Candidate News' that the user may be interested in, based on their 'User's History News'. The recommendations should be made solely on the semantic relevance to the user's interests, not on the order in which the news appears in 'Candidate News'.

# Input
## User's History News
${history}
## Candidate News
${candidate}

# Recommendation Process
1. Analyze 'User's History News' to extract keywords and summarize the user's interests.
2. Group the extracted keywords by meaning to form "Topics" that represent the user's areas of interest.
3. Extract keywords from 'Candidate News' and evaluate how well they match the user's topics of interest.
4. Rank the 'Candidate News' based on their semantic relevance to the user's interests, disregarding their order of appearance.

# Output Format
Summarize the user's interests and rank candidate news according to their relevance to the user's interest in the format: "Ranked news: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>".

"""
