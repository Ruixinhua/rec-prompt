meta_instruction = """You should generate an improved prompt template based on the provided information, which consists of two parts: 
- the current prompt template between "# Current Prompt Template Begin" and "# Current Prompt Template End" and the previous best prompt template between "# Previous Best Prompt Template Begin" and "# Previous Best Prompt Template End".
- some samples with required fields by the prompt template and observation results between "# Sample Begin" and "# Sample End". 
You should provide an enhanced prompt template between "# Prompt Template Begin" and "# Prompt Template End". Use "${history}" as the placeholder for the user's history news and "${candidate}" as the placeholder for the candidate news.
"""
# and a monitor that records the current performance and previous best performance with the corresponding prompt template under "# Monitor". You should provide an enhanced prompt template between "# Prompt Template Begin" and "# Prompt Template End".
guide_instruction_complex = """Generate a better prompt that guides the recommender in making recommendations. The generated prompt must be placed between "# Prompt Template Begin" and "# Prompt Template End" and must include the placeholders "${history}" under "## User's History News" and "${candidate}" under "## Candidate News". Observe the differences between "# Best Prompt Template" and "# Current Prompt Template". If "# Best Prompt Template" and "# Current Prompt Template" are different, analyze why "# Best Prompt Template" is better than "# Current Prompt Template", and on this basis, improve "# Best Prompt Template" to achieve better recommendation results. Do not directly copy sentences from the "# Current Prompt Template" and "# Best Prompt Template". 
"""
guide_instruction_naive = """The generated prompt must be placed between "# Prompt Template Begin" and "# Prompt Template End" and must include the placeholders "${history}" under "## User's History News" and "${candidate}" under "## Candidate News".
"""
non_repeat = """Don't directly copy and paste the "# Current Prompt Template" and "# Best Prompt Template" as the generated prompt."""

prompt_temp = """# Current Prompt Template Begin
${prompt_temp}
# Current Prompt Template End
"""

best_prompt_temp = """# Previous Best Prompt Template Begin
${best_prompt_temp}
# Previous Best Prompt Template End
"""

sample_temp = """# Sample Begin
## Prompt for the Recommender 
${full_prompt}
## Recommender's Answer
${answer}
## User's Click
${click}
# Sample End
"""

observation_instruction_naive = """Focus on the given sample and analyze the content of the prompt for the recommender and the recommender's answer based on the user's click. Observations can focus on the following aspects:
- Whether the recommender's answer correctly extracts the keywords from the user's history news.
- Whether the keywords extracted from the recommender's answer accurately summarise the topics of interest to the user.
- Whether the recommender's answer correctly extracts keywords from the candidate news and correctly matches the user's interests.
You should generate an enhanced prompt template based on the above observations, detailing the task requirements so that the recommender can answer using the guidance given above. Use "${history}" as the placeholder for the "User's History News" and "${candidate}" as the placeholder for the "Candidate News". Based on the observations, you can enhance the prompt by adding a description of the recommendation task, the breakdown of the recommendation process, and the constraints on the final output format so that the recommender can make corresponding recommendations based on semantics only."""

observation_instruction_complex = """Focus on the given sample and analyze the content of the prompt for the recommender and the recommender's answer based on the user's click. Observations can focus on the following aspects:
- Whether the recommender's answer correctly extracts the keywords from the user's history news.
- Whether the keywords extracted from the recommender's answer accurately summarise the topics of interest to the user.
- Whether the recommender's answer correctly extracts keywords from the candidate news and correctly matches the user's interests.
Based on the observations, you can enhance the prompt by adding a description of the recommendation task, the breakdown of the recommendation process, and the constraints on the final output format so that the recommender can make corresponding recommendations based on semantics only. Here are some suggestions for enhancing the prompt:
- Under "# Input", you can use "${history}" as the placeholder for the user's history news and "${candidate}" as the placeholder for the candidate news.
- Under "# Task Description", you can add a description of the news recommendation task, such as 'User's History News' being the news the user has interacted with previously and 'Candidate News' being news the user may be interested in. The recommendation should solely rely on the user's interests, not the order in which the news appears in 'Candidate News'. This task selects the news most relevant to the user's interests.
- Under "# Recommendation Process", a detailed recommendation process can be added, such as first analyzing and summarizing the user's interests. Keywords are extracted from 'User's History News' and then grouped by meaning, with semantically similar words representing related concepts called a "Topic". These keywords can summarize and infer topics the user is interested in. Then, extract keywords from 'Candidate News' and analyze how well the keywords extracted from each Candidate News match the user's topics of interest. Again, it emphasized that the recommendation should not be affected by the position of the news in the  'Candidate News' but only by the matching relationship between the candidate news and the user's interests.
- Under "# Output Format", define the output format. Summarize the user's interest and rank candidate news according to their relevance to the user's interest in the format:  "Candidate news ranked solely by relevance to the user's interests: <START>C#, C#, C#, C#, C#, C#, C#, C#, C#, C#<END>". The model must also summarize the user's interests and explain the recommendation results."""

# - Under "# Output Format", define the output format. Summarize the user's interest and rank candidate news according to their relevance to the user's interest. The output should first rank candidate news based on the user's history news and then provide explanations. The explanation should accurately and comprehensively summarize the topics of the user's interest. Then, you should also infer the candidates' topics and determine which candidates are relevant to the user's interest."""

observation_instruction_refined = """Focus on the given sample and analyze the content of the prompt for the recommender and the recommender's answer based on the user's click. Observations can focus on the following aspects:
- Whether the recommender's answer correctly extracts the keywords from the user's history news.
- Whether the keywords extracted from the recommender's answer accurately summarise the topics of interest to the user.
- Whether the recommender's answer correctly extracts keywords from the candidate news and correctly matches the user's interests.
Based on the observations, you can enhance the prompt by adding a description of the recommendation task, the breakdown of the recommendation process, and the constraints on the final output format so that the recommender can make corresponding recommendations based on semantics only. Here are some suggestions for enhancing the prompt:
- First, you can ask the model to "think step by step and recommend candidate news articles".
- Under "# Input", you can use "${history}" as the placeholder for the user's history news and "${candidate}" as the placeholder for the candidate news.
- Under "# Task Description", you can add a description of the news recommendation task, such as 'User's History News' being the news the user has interacted with previously and 'Candidate News' being news the user may be interested in. The recommendation should solely rely on the user's interests, not the order in which the news appears in 'Candidate News'. This task selects the news most relevant to the user's interests.
- Under "# Recommendation Process", a detailed recommendation process can be added, such as first analyzing and summarizing the user's interests. Keywords are extracted from 'User's History News' and then grouped by meaning, with semantically similar words representing related concepts called a "Topic". These keywords can summarize and infer topics the user is interested in. Then, extract keywords from 'Candidate News' and analyze how well the keywords extracted from each Candidate News match the user's topics of interest. Again, it emphasized that the recommendation should not be affected by the position of the news in the  'Candidate News' but only by the matching relationship between the candidate news and the user's interests.
- Under "# Output Format", define the output format. Summarize the user's interest and rank candidate news according to their relevance to the user's interest in the JSON format."""
