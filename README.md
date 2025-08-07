# uhealth

The UHC Network Affordability team is focused on getting our members the right care for the best possible price in the most convenient manner.

The task at hand is to perform an exploratory data analysis (EDA) on publicly available claim data for Medicare members in the US. Medicare is the government healthcare program for US citizens aged 65+. 

The aim is to understand how the cost of treating certain chronic conditions varies across different providers.

Guidelines 

•	Please complete this task and share output, code & any relevant materials (slides, workbooks etc.) by 10am Monday 4th August.
•	Please take the time to understand the business problem and to develop a thoughtful solution, we suggest taking a few hours to complete the task.
•	If there are parts of this EDA that you would do differently, or with more time try something else, please discuss this during the interview. We can also discuss tools, packages, libraries used.
•	However, the interview panel will be most interested in seeing the results of your analysis, and how costs of different illnesses or providers compare. Please be prepared to present your results to a non-technical audience.

1.	Data

The data is on the CMS (Medicare) website.

Member Benefit Data
https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2009_Beneficiary_Summary_File_Sample_20.zip

Outpatient Claim Data
https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2008_to_2010_Outpatient_Claims_Sample_20.zip

User Documentation
https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/SynPUF_DUG.pdf

Additional Info
https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs

2.	Data Cleaning

The beneficiary summary file has several chronic illness columns for each member. These are Boolean fields.

•	Convert these columns into a single categorical variable, concatenating multiple true diagnoses.
•	If a member has three or more chronic conditions, categorise these as “Multiple”.
•	Join claims & benefit data.

3.	Basic Summaries

•	What is the distribution of races?
•	What is the most common chronic illness combination?
•	Which chronic illness combination has the total highest cost?
•	Which chronic illness combination has the highest cost per member?

4.	Benchmarking

The aim here is to understand the distribution of cost across providers treating members with these chronic illnesses. Benchmarking providers across types of care is often a helpful starting point to begin working with areas of excessive cost.

•	For each provider (use AT_PHYSN_NPI) & chronic illness, calculate the cost per member.
•	For each chronic illness combination, represent the distribution of costs per provider.
•	How does this change if we filter out cases where a given Chronic Illness & Provider NPI combination only has one member?
•	Which providers are consistently expensive across chronic illnesses they treat?

python explore_analytics.py 
python explore_analytics_cluster.py
python explore_cluster.py 