rec_cot_3t3 = """## Task Description
The news recommendation task involves recommending candidate news articles based on the user's interests. 'User's History News' represents the news the user has interacted with previously, and 'Candidate News' represents news the user may be interested in. The recommendation should solely rely on the user's interests, not the order in which the news appears in 'Candidate News'. This task selects the news most relevant to the user's interests.

## Recommendation Process
1. Analyze and summarize the user's interests: Extract keywords from 'User's History News' and group them by meaning, with semantically similar words representing related concepts called a "Topic". These keywords can summarize and infer topics the user is interested in.
2. Extract keywords from 'Candidate News': Analyze how well the keywords extracted from each Candidate News match the user's topics of interest. The recommendation should not be affected by the position of the news in the 'Candidate News' but only by the matching relationship between the candidate news and the user's interests.

## Input
### User's History News
${history}
### Candidate News
${candidate}

## Output Format
Summarize the user's interest and rank candidate news according to their relevance to the user's interest in the format: "Candidate news ranked solely by relevance to the user's interests: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>".
"""

rec_cot_pure_3t3 = """## Task Description
The news recommendation task involves recommending candidate news articles based on the user's interests. 'User's History News' represents the news the user has interacted with previously, and 'Candidate News' represents news the user may be interested in. The recommendation should solely rely on the user's interests, not the order in which the news appears in 'Candidate News'. This task selects the news most relevant to the user's interests.

## Recommendation Process
1. Analyze and summarize the user's interests: Extract keywords from 'User's History News' and group them by meaning, with semantically similar words representing related concepts called a "Topic". These keywords can summarize and infer topics the user is interested in.
2. Extract keywords from 'Candidate News': Analyze how well the keywords extracted from each Candidate News match the user's topics of interest. The recommendation should not be affected by the position of the news in the 'Candidate News' but only by the matching relationship between the candidate news and the user's interests.

## Input
### User's History News
${history}
### Candidate News
${candidate}

## Output Format
Rank candidate news based on the user's history news with the given format.
"""

rec_cot_4t3 = """Based on the user's news history, analyze and recommend candidate news articles that align with the user's interests. Use a step-by-step approach to ensure the recommendations are semantically related to the user's preferences.

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

rec_cot_pure_4t3 = """Based on the user's news history, analyze and recommend candidate news articles that align with the user's interests. Use a step-by-step approach to ensure the recommendations are semantically related to the user's preferences.

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
Rank candidate news based on the user's history news with the given format.
"""

rec_cot_cat_4t3 = """Guide the recommender to make news recommendations based on the user's history news. The recommender should analyze the user's interests and match them with the candidate news articles.

# Task Description
The goal is to recommend news articles from the candidate list that align with the user's historical interests. 'User's History News' represents the news articles the user has previously shown interest in, while 'Candidate News' includes potential articles of interest. The recommendation should be based solely on the semantic relevance to the user's interests, disregarding the order of appearance in 'Candidate News'.

# Recommendation Process
1. Analyze 'User's History News' to extract keywords and summarize the user's interests into topics.
2. Group semantically similar keywords to form a "Topic" representing a specific area of interest.
3. Extract keywords from each 'Candidate News' article.
4. Determine the semantic match between the keywords from 'Candidate News' and the user's topics of interest.
5. Ensure that the recommendation is not influenced by the position of the news in 'Candidate News', but solely by the relevance to the user's interests.

# Input
## User's History News
${history}
## Candidate News
${candidate}

# Output Format
Summarize the user's interests and rank the candidate news according to their relevance to these interests. Use the following format for the output: "Candidate news ranked solely by relevance to the user's interests: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>".

"""

rec_io_4t4 = """# Input:
## User's History News:
${history}
## Candidate News:
${candidate}

# Task Description:
The task is to recommend news articles to a user based on their history of news interactions. 'User's History News' represents articles the user has previously shown interest in, while 'Candidate News' comprises potential articles that may align with the user's interests. The goal is to identify and rank the candidate news articles that are most relevant to the user's demonstrated interests, disregarding the order in which they are presented in 'Candidate News'.

# Recommendation Process:
1. Analyze 'User's History News' to extract and summarize the user's interests. Identify keywords and group them by meaning to form "Topics" that represent the user's areas of interest.
2. Extract keywords from each 'Candidate News' article and evaluate how well these keywords align with the user's Topics of interest.
3. Ensure that the recommendation is not influenced by the sequence of the news in 'Candidate News' but solely by the semantic relevance of each candidate article to the user's interests.

# Output Format:
Rank the candidate news articles based on their relevance to the user's interests in the following format: "Candidate news ranked solely by relevance to the user's interests: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>".
You should output the ranked list FIRST and then explain the results."""

rec_io_cat_pure_4t3 = """# Input:
## User's History News:
${history}
## Candidate News:
${candidate}

# Task Description:
The task is to recommend news articles to the user based on their history of interactions. 'User's History News' represents the articles the user has previously shown interest in, while 'Candidate News' includes potential articles that may be of interest to the user. The goal is to identify and rank the candidate news articles that are most relevant to the user's demonstrated interests, without being influenced by the order in which they are presented in 'Candidate News'.

# Recommendation Process:
1. Analyze 'User's History News' to extract and summarize the user's interests. Identify keywords and group them by meaning to form "Topics" that represent the user's areas of interest.
2. Extract keywords from 'Candidate News' and evaluate how well they align with the user's topics of interest.
3. Rank the candidate news articles based on the semantic relevance of their content to the user's interests. The position of the news in 'Candidate News' should not influence the ranking; the focus should be on the semantic match between the candidate news and the user's interests.

# Output Format:
Rank candidate news according to their relevance to the user's interests with the given format."""

rec_io_pure_4t4 = """# Input:
## User's History News:
${history}
## Candidate News:
${candidate}

# Task Description:
The task is to recommend news articles to a user based on their history of news interactions. 'User's History News' represents articles the user has previously shown interest in, while 'Candidate News' comprises potential articles that may align with the user's interests. The goal is to identify and rank the candidate news articles that are most relevant to the user's demonstrated interests, disregarding the order in which they are presented in 'Candidate News'.

# Recommendation Process:
1. Analyze 'User's History News' to extract and summarize the user's interests. Identify keywords and group them by meaning to form "Topics" that represent the user's areas of interest.
2. Extract keywords from each 'Candidate News' article and evaluate how well these keywords align with the user's Topics of interest.
3. Ensure that the recommendation is not influenced by the sequence of the news in 'Candidate News' but solely by the semantic relevance of each candidate article to the user's interests.

# Output Format:
Rank the candidate news articles based on their relevance to the user's interests with the given format."""

rec_cot_pure_4t4 = """Based on the user's news history, think step by step and recommend candidate news articles.
# Input:
## User's History News:
${history}
## Candidate News:
${candidate}

# Task Description:
The task is to recommend news articles to a user based on their history of news interactions. 'User's History News' represents articles the user has previously shown interest in, while 'Candidate News' comprises potential articles that may align with the user's interests. The goal is to identify and rank the candidate news articles that are most relevant to the user's demonstrated interests, disregarding the order in which they are presented in 'Candidate News'.

# Recommendation Process:
1. Analyze 'User's History News' to extract and summarize the user's interests. Identify keywords and group them by meaning to form "Topics" that represent the user's areas of interest.
2. Extract keywords from each 'Candidate News' article and evaluate how well these keywords align with the user's Topics of interest.
3. Ensure that the recommendation is not influenced by the sequence of the news in 'Candidate News' but solely by the semantic relevance of each candidate article to the user's interests.

# Output Format:
Rank the candidate news articles based on their relevance to the user's interests with the given format."""

rec_cot_cat_pure_4t3 = """Guide the recommender to make news recommendations based on the user's history news. The recommender should analyze the user's interests and match them with the candidate news articles.

# Task Description
The goal is to recommend news articles from the candidate list that align with the user's historical interests. 'User's History News' represents the news articles the user has previously shown interest in, while 'Candidate News' includes potential articles of interest. The recommendation should be based solely on the semantic relevance to the user's interests, disregarding the order of appearance in 'Candidate News'.

# Recommendation Process
1. Analyze 'User's History News' to extract keywords and summarize the user's interests into topics.
2. Group semantically similar keywords to form a "Topic" representing a specific area of interest.
3. Extract keywords from each 'Candidate News' article.
4. Determine the semantic match between the keywords from 'Candidate News' and the user's topics of interest.
5. Ensure that the recommendation is not influenced by the position of the news in 'Candidate News', but solely by the relevance to the user's interests.

# Input
## User's History News
${history}
## Candidate News
${candidate}

# Output Format
Rank candidate news according to their relevance to these interests with the given format."""

rec_cot_cat_2_pure_4t3 = """Based on the user's news history, analyze and recommend candidate news articles that align with the user's interests. Use a step-by-step approach to ensure the recommendations are semantically related to the user's preferences.

# Task Description
The goal is to recommend news articles from the 'Candidate News' that the user is most likely to be interested in, based on their 'User's History News'. The 'User's History News' consists of articles the user has previously interacted with, while the 'Candidate News' includes potential articles of interest. The recommendation should be based solely on the semantic relevance to the user's interests, not on the order of appearance in the 'Candidate News'.

# Recommendation Process
1. Analyze 'User's History News' to extract and summarize the user's interests. Identify keywords and group them by meaning to form topics that represent the user's areas of interest.
2. Extract keywords from each 'Candidate News' article and evaluate how well these keywords match the user's topics of interest.
3. Ensure that the recommendation is not influenced by the position of the news in the 'Candidate News' list but solely by the semantic relationship between the candidate news content and the user's interests.

# Input
## User's History News
${history}
## Candidate News
${candidate}

# Output Format
Rank the candidate news according to their relevance to the user's interests with the given format."""

rec_io_3t3 = """# Input:
## User's History News:
${history}
## Candidate News:
${candidate}
# Task Description:
This task involves recommending news articles based on the user's interests. 'User's History News' represents the news the user has interacted with previously, and 'Candidate News' represents news the user may be interested in. The recommendation should solely rely on the user's interests, not the order in which the news appears in 'Candidate News'. This task selects the news most relevant to the user's interests.
# Recommendation Process:
1. Analyze and summarize the user's interests by extracting keywords from 'User's History News'.
2. Group the extracted keywords by meaning, with semantically similar words representing related concepts called a "Topic".
3. Extract keywords from 'Candidate News' and analyze how well the keywords extracted from each Candidate News match the user's topics of interest.
4. Rank the candidate news solely based on their relevance to the user's interests, without considering the order in which they appear in 'Candidate News'.
# Output Format:
Summarize the user's interest and rank candidate news according to their relevance to the user's interest in the format: "Candidate news ranked solely by relevance to the user's interests: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>".
"""

rec_io_4t3 = """# Input:
## User's History News:
${history}
## Candidate News:
${candidate}

# Task Description:
The task is to recommend news articles to a user based on their history of news interactions. 'User's History News' consists of articles the user has previously shown interest in, while 'Candidate News' includes potential articles that may be of interest to the user. The goal is to identify and rank the candidate news articles that are most relevant to the user's demonstrated interests. The recommendation should be based solely on the semantic relevance of the content to the user's interests, without consideration for the order in which the candidate news is presented.

# Recommendation Process:
1. Analyze 'User's History News' to extract keywords and summarize the user's interests into distinct topics.
2. Group semantically similar keywords to form a "Topic" that represents a specific area of interest.
3. Extract keywords from each 'Candidate News' article to determine the thematic content.
4. Match the keywords from 'Candidate News' with the user's interest topics to assess relevance.
5. Ensure that the recommendation is not influenced by the position of the news in 'Candidate News', but solely by the semantic relationship between the candidate news and the user's interests.

# Output Format:
Summarize the user's interests and rank the candidate news according to their relevance to these interests. The ranked list should be presented in the following format: "Candidate news ranked solely by relevance to the user's interests: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>".
"""

rec_io_cat_4t3 = """# Input:
## User's History News:
${history}
## Candidate News:
${candidate}

# Task Description:
The task is to recommend news articles to the user based on their history of interactions. 'User's History News' represents the articles the user has previously shown interest in, while 'Candidate News' includes potential articles that may be of interest to the user. The goal is to identify and rank the candidate news articles that are most relevant to the user's demonstrated interests, without being influenced by the order in which they are presented in 'Candidate News'.

# Recommendation Process:
1. Analyze 'User's History News' to extract and summarize the user's interests. Identify keywords and group them by meaning to form "Topics" that represent the user's areas of interest.
2. Extract keywords from 'Candidate News' and evaluate how well they align with the user's topics of interest.
3. Rank the candidate news articles based on the semantic relevance of their content to the user's interests. The position of the news in 'Candidate News' should not influence the ranking; the focus should be on the semantic match between the candidate news and the user's interests.

# Output Format:
Rank candidate news according to their relevance to the user's interests in the format: "Candidate news ranked solely by relevance to the user's interests: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>".

"""

rec_cot_cat_4t4 = """Based on the user's news history, analyze the user's interests and recommend candidate news articles that align with those interests. Use a thoughtful, step-by-step approach to determine the most relevant articles from the candidate list.

# Task Description
The goal is to provide personalized news recommendations to the user. 'User's History News' consists of news articles and their corresponding categories that the user has previously shown interest in. 'Candidate News' comprises news articles and categories that the user may find appealing. The recommendation should prioritize the user's demonstrated interests, as inferred from the history, rather than the sequence of news items in 'Candidate News'.

# Recommendation Process
1. Review the 'User's History News' to identify patterns in topics, categories, and specific interests.
2. Compare these interests with the topics and categories of 'Candidate News'.
3. Rank the 'Candidate News' articles in order of relevance to the user's identified interests, ensuring that the most aligned articles are recommended first.

# Input
## User's History News
${history}
## Candidate News
${candidate}

# Output Format
Provide a ranked list of candidate news articles based on their relevance to the user's history news. Use the following format for the recommendation:

{
    "ranking": ["C#", "C#", ...] // Replace C# with the candidate news identifiers in order of relevance.
}

"""