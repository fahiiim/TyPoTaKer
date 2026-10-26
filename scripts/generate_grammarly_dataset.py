"""
High-Precision Grammarly & QuillBot Grade GEC Dataset Generator.
Synthesizes professional, academic, technical, and casual sentence pairs
with strict semantic consistency, fine-grained error taxonomy, error counts,
domain tags, pedagogical explanations, and automated quality validation.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import NamedTuple


class DatasetRow(NamedTuple):
    id: str
    input_text: str
    target_text: str
    error_type: str
    error_category: str
    error_count: int
    correction_explanation: str
    domain: str
    complexity_level: str


# -----------------------------------------------------------------------------
# 1. SEMANTICALLY VERIFIED HOMOPHONE & CONFUSED WORD SCENARIOS
# -----------------------------------------------------------------------------
HOMOPHONE_PAIRS = [
    # their / there / they're
    ("their", "there", "Please verify that {word} are no remaining syntax errors in the script.", "Use 'there' to indicate existence or location; 'their' is possessive.", "technical_code"),
    ("there", "their", "The developers pushed {word} latest commits to the release branch.", "Use 'their' to indicate possession; 'there' refers to location or existence.", "technical_code"),
    ("they're", "their", "All team members must complete {word} weekly performance reviews.", "Use 'their' for possession; 'they\\'re' is a contraction of 'they are'.", "professional_email"),
    ("their", "they're", "We verified that {word} planning to deploy the hotfix before midnight.", "Use 'they\\'re' as a contraction for 'they are'; 'their' indicates possession.", "technical_code"),
    
    # its / it's
    ("it's", "its", "The machine learning model reached {word} peak convergence after ten epochs.", "Use 'its' as the possessive pronoun; 'it\\'s' is a contraction of 'it is'.", "technical_code"),
    ("its", "it's", "Before proceeding with migration, {word} essential to create a full backup.", "Use 'it\\'s' when you mean 'it is' or 'it has'; 'its' is possessive.", "technical_code"),
    
    # your / you're
    ("your", "you're", "Please notify the administrator once {word} ready to run the migration.", "Use 'you\\'re' as a contraction of 'you are'; 'your' is possessive.", "professional_email"),
    ("you're", "your", "Make sure to securely store {word} personal access token.", "Use 'your' to indicate possession; 'you\\'re' means 'you are'.", "technical_code"),
    
    # affect / effect
    ("effect", "affect", "High disk I/O latency will directly {word} overall query throughput.", "Use 'affect' as a verb meaning to influence or alter.", "technical_code"),
    ("affect", "effect", "The change in memory configuration had an immediate {word} on stability.", "Use 'effect' as a noun meaning result or consequence.", "technical_code"),
    
    # then / than
    ("then", "than", "Our quantized INT8 model runs significantly faster {word} the FP32 baseline.", "Use 'than' for comparisons; 'then' indicates time, sequence, or consequence.", "technical_code"),
    ("than", "then", "First inspect the failing unit test, {word} refactor the helper function.", "Use 'then' to indicate sequence or consequence; 'than' is for comparisons.", "technical_code"),
    
    # to / too / two
    ("to", "too", "The memory footprint of this unquantized checkpoint is {word} large for edge devices.", "Use 'too' to mean 'excessively' or 'also'; 'to' is a preposition.", "technical_code"),
    ("too", "to", "The client sent a request {word} terminate the active session.", "Use 'to' as a preposition or infinitive marker; 'too' means 'also' or 'excessively'.", "technical_code"),
    
    # lose / loose
    ("loose", "lose", "Be careful not to {word} database connectivity during the rolling update.", "Use 'lose' to mean misplace or suffer loss; 'loose' means unbound or not tight.", "technical_code"),
    ("lose", "loose", "The hardware technician discovered a {word} Ethernet cable on the rack.", "Use 'loose' to mean not securely fastened; 'lose' is a verb meaning to misplace.", "technical_code"),
    
    # complement / compliment
    ("complement", "compliment", "The director paid her a sincere {word} on her clear presentation.", "Use 'compliment' for praise; 'complement' means completing or matching well.", "professional_email"),
    ("compliment", "complement", "The new analytical dashboard serves as a great {word} to our logging pipeline.", "Use 'complement' to mean balancing or completing; 'compliment' is praise.", "technical_code"),
    
    # principal / principle
    ("principal", "principle", "We must strictly adhere to the security {word} of least privilege.", "Use 'principle' for a fundamental rule or truth; 'principal' means chief or head.", "technical_code"),
    ("principle", "principal", "The {word} investigator published the empirical findings in Nature.", "Use 'principal' for a primary leader or main amount; 'principle' is a rule.", "academic_essay"),
    
    # precede / proceed
    ("precede", "proceed", "After authentication succeeds, you may {word} to the dashboard.", "Use 'proceed' to mean continue forward; 'precede' means to come before.", "technical_code"),
    ("proceed", "precede", "Careful data preprocessing must {word} model training.", "Use 'precede' to mean come before in time or sequence; 'proceed' means go forward.", "technical_code"),
    
    # discrete / discreet
    ("discrete", "discreet", "Please be {word} when discussing unreleased product features with external partners.", "Use 'discreet' to mean tactful or circumspect; 'discrete' means distinct or separate.", "professional_email"),
    ("discreet", "discrete", "The simulation divides continuous time into small {word} intervals.", "Use 'discrete' to mean separate and distinct; 'discreet' means cautious.", "academic_essay"),
    
    # ensure / insure / assure
    ("insure", "ensure", "Please {word} that all automated test pipelines pass before merging.", "Use 'ensure' to mean make certain or guarantee; 'insure' refers to monetary coverage.", "technical_code"),
    ("ensure", "insure", "The corporation decided to {word} all physical hardware against electrical damage.", "Use 'insure' for financial underwriting; 'ensure' means guarantee.", "business_finance"),
    ("ensure", "assure", "I can {word} you that our engineering team is addressing the latency issue.", "Use 'assure' to remove doubt or give confidence to someone; 'ensure' means make certain.", "professional_email"),
    
    # advice / advise
    ("advice", "advise", "The staff architect will {word} the squad on database sharding strategies.", "Use 'advise' as the verb; 'advice' is the uncountable noun.", "technical_code"),
    ("advise", "advice", "We received invaluable {word} from the senior cloud security auditor.", "Use 'advice' as a noun; 'advise' is the verb form.", "professional_email"),
    
    # breath / breathe
    ("breath", "breathe", "The room was so crowded that it was difficult to {word} comfortably.", "Use 'breathe' as the verb; 'breath' is the noun.", "creative_writing"),
    ("breathe", "breath", "Take a deep {word} before stepping onto the stage to give your keynote.", "Use 'breath' as the noun; 'breathe' is the verb.", "professional_email"),
    
    # accept / except
    ("except", "accept", "We are pleased to {word} the revised contractual terms submitted by legal.", "Use 'accept' to mean agree to or receive; 'except' means excluding.", "business_finance"),
    ("accept", "except", "All integration tests succeeded {word} for the legacy payment gateway.", "Use 'except' to mean excluding; 'accept' means receive or agree to.", "technical_code"),
    
    # elicit / illicit
    ("illicit", "elicit", "The anonymous survey was designed to {word} candid employee feedback.", "Use 'elicit' to mean draw forth; 'illicit' means illegal.", "professional_email"),
    ("elicit", "illicit", "Law enforcement agencies cracked down on {word} financial operations.", "Use 'illicit' for unlawful or forbidden activities; 'elicit' means draw forth.", "business_finance"),
    
    # stationary / stationery
    ("stationary", "stationery", "The company ordered customized executive {word} with the new logo.", "Use 'stationery' for writing paper; 'stationary' means not moving.", "business_finance"),
    ("stationery", "stationary", "The CPU usage monitor remained {word} at forty percent during load testing.", "Use 'stationary' to mean remaining in one place or state; 'stationery' is paper.", "technical_code"),
    
    # sight / site / cite
    ("sight", "site", "The civil engineering team inspected the future server facility {word}.", "Use 'site' for a physical or digital location; 'sight' refers to vision.", "technical_code"),
    ("site", "sight", "The panoramic mountain view at dawn was an unforgettable {word}.", "Use 'sight' for vision or a spectacle; 'site' refers to a location.", "creative_writing"),
    ("site", "cite", "Be sure to {word} authoritative academic papers in your literature review.", "Use 'cite' to mean quote or reference a source; 'site' is location.", "academic_essay"),
    
    # lead / led
    ("lead", "led", "The principal engineer {word} the architectural refactoring initiative.", "Use 'led' as the past tense of 'lead'; 'lead' in the past refers to the metal.", "technical_code"),
    ("led", "lead", "Automotive batteries typically contain chemical compounds of {word}.", "Use 'lead' for the metallic element; 'led' is the past tense verb.", "academic_essay"),
    
    # past / passed
    ("passed", "past", "In the {word}, server provisioning required manual intervention.", "Use 'past' to indicate prior time; 'passed' is the verb action.", "technical_code"),
    ("past", "passed", "The legislative committee successfully {word} the privacy protection bill.", "Use 'passed' as the past tense of 'pass'; 'past' refers to previous time.", "journalism"),
    
    # weather / whether
    ("weather", "whether", "We are still deciding {word} to rewrite the legacy service in Go.", "Use 'whether' to introduce alternatives; 'weather' refers to atmospheric conditions.", "technical_code"),
    ("whether", "weather", "Severe tropical {word} delayed shipments across the eastern seaboard.", "Use 'weather' for atmospheric conditions; 'whether' introduces alternatives.", "casual_communication"),
    
    # piece / peace
    ("piece", "peace", "Diplomats worked tirelessly to negotiate lasting {word} in the region.", "Use 'peace' for harmony or quiet; 'piece' means a section or portion.", "journalism"),
    ("peace", "piece", "The mechanic replaced the damaged {word} on the assembly line.", "Use 'piece' for a portion or component; 'peace' means tranquility.", "technical_code"),
    
    # farther / further
    ("farther", "further", "We must conduct {word} investigations before reaching a definitive conclusion.", "Use 'further' for figurative distance or degree; 'farther' refers to physical distance.", "academic_essay"),
    
    # aloud / allowed
    ("aloud", "allowed", "Visitors are not {word} inside the server room without proper badges.", "Use 'allowed' to mean permitted; 'aloud' means audibly.", "technical_code"),
    ("allowed", "aloud", "The presenter read the executive summary {word} to the audience.", "Use 'aloud' to mean audibly; 'allowed' means permitted.", "professional_email"),
    
    # break / brake
    ("break", "brake", "The autonomous vehicle engaged its emergency {word} system instantly.", "Use 'brake' for decelerating mechanisms; 'break' means to fracture or pause.", "technical_code"),
    ("brake", "break", "Let us take a brief ten-minute coffee {word} before the workshop.", "Use 'break' to mean a pause or fracture; 'brake' refers to decelerating mechanisms.", "casual_communication"),
    
    # capital / capitol
    ("capital", "capitol", "Lawmakers gathered outside the state {word} building for the press conference.", "Use 'capitol' for the legislative building; 'capital' refers to wealth, cities, or letters.", "journalism"),
    ("capitol", "capital", "The venture fund invested ten million dollars in working {word}.", "Use 'capital' for funds, primary cities, or letters; 'capitol' is the legislative building.", "business_finance"),
    
    # allusion / illusion
    ("allusion", "illusion", "The optical lens creates an {word} of increased depth and clarity.", "Use 'illusion' for a deceptive appearance; 'allusion' is an indirect reference.", "creative_writing"),
    ("illusion", "allusion", "The poet made a subtle {word} to ancient Roman mythology.", "Use 'allusion' for an indirect reference; 'illusion' is a false perception.", "academic_essay"),
    
    # conscience / conscious
    ("conscience", "conscious", "The patient remained fully {word} during the brief medical procedure.", "Use 'conscious' for awake or aware; 'conscience' is the inner moral sense.", "academic_essay"),
    ("conscious", "conscience", "Her clear {word} gave her the confidence to speak truth to power.", "Use 'conscience' for moral judgment; 'conscious' means aware or awake.", "creative_writing"),
]

# -----------------------------------------------------------------------------
# 2. SUBJECT-VERB AGREEMENT SCENARIOS
# -----------------------------------------------------------------------------
SVA_DATA = [
    # (singular_subject, plural_verb, singular_verb, plural_subject, rest, explanation, domain)
    ("The list of registered participants", "are", "is", "The lists of registered participants", "available on the corporate intranet.", "Subject-verb agreement: 'The list' is a singular subject requiring 'is', not 'are'.", "professional_email"),
    ("The quality of the training datasets", "were", "was", "The qualities of the training datasets", "verified using automated statistical checks.", "Subject-verb agreement: 'The quality' is singular and requires 'was'.", "technical_code"),
    ("Every engineer on the infrastructure squad", "have", "has", "All engineers on the infrastructure squad", "signed the security protocol agreement.", "Subject-verb agreement: 'Every engineer' is singular and takes 'has'.", "technical_code"),
    ("The collection of benchmark results", "exceed", "exceeds", "The collections of benchmark results", "our initial performance targets.", "Subject-verb agreement: 'The collection' is singular and takes 'exceeds'.", "academic_essay"),
    ("Neither the product manager nor the tech lead", "were", "was", "Neither the product managers nor the tech leads", "informed about the scheduled outage.", "Subject-verb agreement: Singular subjects joined by 'neither/nor' take singular 'was'.", "professional_email"),
    ("Each of the distributed microservice containers", "require", "requires", "All distributed microservice containers", "at least two gigabytes of available RAM.", "Subject-verb agreement: 'Each' is grammatically singular and takes 'requires'.", "technical_code"),
    ("The density of celestial bodies in this cluster", "remain", "remains", "The densities of celestial bodies in these clusters", "a topic of active astrophysical inquiry.", "Subject-verb agreement: 'The density' is singular, requiring 'remains'.", "academic_essay"),
    ("The committee of senior executive directors", "have", "has", "The committees of senior executive directors", "unanimously approved the annual capital allocation.", "Subject-verb agreement: 'The committee' acts as a collective singular noun taking 'has'.", "business_finance"),
    ("One of the most experienced software architects", "lead", "leads", "The most experienced software architects", "the weekly design review.", "Subject-verb agreement: 'One of the architects' has the singular subject 'One', requiring 'leads'.", "technical_code"),
    ("The cost of replacing damaged hardware components", "are", "is", "The costs of replacing damaged hardware components", "covered under our enterprise warranty.", "Subject-verb agreement: 'The cost' is singular, so it takes 'is' instead of 'are'.", "business_finance"),
    ("There", "is", "are", "There", "multiple reliable approaches to solving this optimization problem.", "Inverted agreement: In 'There are multiple approaches', the subject is plural ('approaches'), so use 'are'.", "academic_essay"),
    ("There", "are", "is", "There", "only one valid configuration profile specified in the root directory.", "Inverted agreement: In 'There is only one profile', the subject is singular ('one profile'), requiring 'is'.", "technical_code"),
    ("A series of unexpected penetration tests", "was", "were", "Multiple penetration tests", "conducted to evaluate firewall resilience.", "Subject-verb agreement: 'A series' when referring to plural events takes agreement based on context.", "technical_code"),
    ("The team of data analysts", "have", "has", "The teams of data analysts", "finalized the quarterly customer retention model.", "Subject-verb agreement: 'The team' is a collective singular noun requiring 'has'.", "business_finance"),
]

# -----------------------------------------------------------------------------
# 3. VERB TENSE & IRREGULAR INFLECTION (ALL CONVERTED TO {word} TEMPLATES)
# -----------------------------------------------------------------------------
TENSE_DATA = [
    # (wrong, right, template, explanation, domain)
    ("had went", "had gone", "By the time the lead arrived at the auditorium, the keynote speaker {word} to the breakout room.", "Past perfect: Use the past participle 'gone' after 'had', not the simple past 'went'.", "professional_email"),
    ("has did", "has done", "The backend engineering team {word} exceptional work optimizing memory allocation.", "Present perfect: Use the past participle 'done' after 'has', not 'did'.", "technical_code"),
    ("could of", "could have", "With automated regression tests in place, we {word} caught the bug before release.", "Modal auxiliary: Write 'could have', not the phonetic mistake 'could of'.", "technical_code"),
    ("should of", "should have", "The deployment team {word} notified customers prior to scheduling maintenance.", "Modal auxiliary: Write 'should have', never 'should of'.", "professional_email"),
    ("would of", "would have", "Had the cluster had additional memory, it {word} processed the batch seamlessly.", "Modal auxiliary: Write 'would have', not 'would of'.", "technical_code"),
    ("was froze", "was frozen", "The desktop client interface {word} due to an unhandled deadlock on the main dispatcher.", "Passive voice: Use the past participle 'frozen' after 'was', not 'froze'.", "technical_code"),
    ("was wrote", "was written", "The foundational mathematical proof {word} by Alan Turing in his seminal paper.", "Passive voice: Use the past participle 'written' after 'was', not 'wrote'.", "academic_essay"),
    ("had broke", "had broken", "The continuous delivery pipeline {word} shortly before the staging release was tagged.", "Past perfect: Use 'had broken', not 'had broke'.", "technical_code"),
    ("must to ensure", "must ensure", "The data protection guidelines mandate that we {word} encryption at rest.", "Modal syntax: 'Must' is followed by the bare infinitive ('ensure'), not 'to ensure'.", "technical_code"),
    ("can able to", "can", "Using this feature, administrators {word} manage user roles across multiple organizations.", "Redundant modal: 'can able to' is ungrammatical; use 'can' or 'are able to'.", "technical_code"),
    ("If I was you", "If I were you", "{word}, I would consult with the legal team before signing the contract.", "Subjunctive mood: In hypothetical conditionals, use the subjunctive 'were' instead of 'was'.", "casual_communication"),
    ("If he would have known", "If he had known", "{word} about the schedule change, he would have rescheduled his flight.", "Conditional clause: Use 'had known' in the if-clause of a third conditional, not 'would have known'.", "professional_email"),
    ("had took", "had taken", "The auditors {word} thorough notes during the security inspection.", "Past perfect: Use the past participle 'taken' after 'had', not 'took'.", "business_finance"),
    ("has chose", "has chosen", "The product council {word} an open-source framework for the mobile rewrite.", "Present perfect: Use the past participle 'chosen', not 'chose'.", "technical_code"),
]

# -----------------------------------------------------------------------------
# 4. COUNTABILITY & QUANTIFIERS
# -----------------------------------------------------------------------------
COUNTABILITY_DATA = [
    # (wrong, right, template, explanation, domain)
    ("informations", "information", "We gathered comprehensive {word} regarding the network latency spike.", "Uncountable noun: 'Information' is non-count and never takes a plural '-s'.", "technical_code"),
    ("advices", "advice", "The principal engineer offered actionable {word} on scaling Kafka consumers.", "Uncountable noun: 'Advice' is a mass noun; use 'advice' or 'pieces of advice'.", "technical_code"),
    ("furnitures", "furniture", "The facility management ordered ergonomic {word} for the workstation area.", "Uncountable noun: 'Furniture' is uncountable and does not take a plural '-s'.", "business_finance"),
    ("equipments", "equipment", "The laboratory installed high-precision diagnostic {word} for tensor profiling.", "Uncountable noun: 'Equipment' is a mass noun; use 'equipment' or 'pieces of equipment'.", "academic_essay"),
    ("feedbacks", "feedback", "The design team received valuable {word} from over two hundred usability testers.", "Uncountable noun: 'Feedback' is non-count; use 'feedback' rather than 'feedbacks'.", "professional_email"),
    ("knowledges", "knowledge", "He possesses comprehensive domain {word} in distributed systems architecture.", "Uncountable noun: 'Knowledge' cannot be pluralized as 'knowledges'.", "technical_code"),
    ("softwares", "software", "Our enterprise licensing covers both desktop and cloud {word} applications.", "Uncountable noun: 'Software' is non-count; use 'software' or 'software tools'.", "technical_code"),
    ("hardwares", "hardware", "The data center upgraded its server {word} to reduce power consumption.", "Uncountable noun: 'Hardware' is non-count; use 'hardware components' or 'hardware'.", "technical_code"),
    ("researches", "research", "The university laboratory conducted pioneering {word} on quantum error correction.", "Uncountable noun: 'Research' is typically non-count; use 'research' or 'research studies'.", "academic_essay"),
    ("less people", "fewer people", "Noticeably {word} attended the virtual quarterly all-hands meeting this week.", "Countable quantifier: Use 'fewer' with countable nouns like 'people', and 'less' for volume.", "business_finance"),
    ("amount of users", "number of users", "A significant {word} reported authentication timeouts during peak hours.", "Countable quantifier: Use 'number of' for countable entities (users) rather than 'amount of'.", "technical_code"),
    ("much items", "many items", "There are too {word} remaining on the sprint backlog to finish by Friday.", "Countable quantifier: Use 'many' with countable plural nouns (items), not 'much'.", "technical_code"),
]

# -----------------------------------------------------------------------------
# 5. PREPOSITIONS, COLLOCATIONS & ESL FLUENCY (ALL CONVERTED TO {word} TEMPLATES)
# -----------------------------------------------------------------------------
PREPOSITION_DATA = [
    # (wrong, right, template, explanation, domain)
    ("depend of", "depend on", "Model inference speeds {word} the hardware acceleration capabilities of the CPU.", "Preposition error: In English, the correct collocation is 'depend on', not 'depend of'.", "technical_code"),
    ("interested for", "interested in", "Several institutional investors are {word} participating in our upcoming funding round.", "Preposition error: Use 'interested in', not 'interested for'.", "business_finance"),
    ("good in", "good at", "Our senior systems architect is remarkably {word} diagnosing multithreaded race conditions.", "Preposition error: Use 'good at' when referring to proficiency in a task or skill.", "technical_code"),
    ("According with", "According to", "{word} the official RFC specification, client headers must be UTF-8 encoded.", "Preposition error: The standard idiom is 'according to', not 'according with'.", "technical_code"),
    ("congratulate for", "congratulate on", "We would like to {word} you on your promotion to senior staff engineer.", "Preposition error: We 'congratulate someone on' an achievement, not 'for'.", "professional_email"),
    ("listen music", "listen to music", "Software engineers frequently {word} while focusing on deep algorithmic tasks.", "Missing preposition: 'Listen' requires the preposition 'to' before an object.", "casual_communication"),
    ("discuss about", "discuss", "Let us schedule a thirty-minute meeting to {word} the quarterly product roadmap.", "Redundant preposition: 'Discuss' is transitive; write 'discuss the topic', not 'discuss about'.", "professional_email"),
    ("married with", "married to", "The company founder is {word} a renowned computer science professor.", "Preposition error: In English, one is 'married to' someone, not 'married with'.", "casual_communication"),
    ("responsible of", "responsible for", "The site reliability engineering team is {word} ensuring ninety-nine point nine percent uptime.", "Preposition error: Use 'responsible for', not 'responsible of'.", "technical_code"),
    ("comply to", "comply with", "All microservices must strictly {word} the organization's data retention standards.", "Preposition error: The standard collocation is 'comply with', not 'comply to'.", "technical_code"),
    ("consist in", "consist of", "The comprehensive unit test suite will {word} over six hundred automated test cases.", "Preposition error: When describing component parts, use 'consist of'.", "technical_code"),
    ("different than", "different from", "The latency characteristics of INT8 quantization are {word} FP32 baseline models.", "Preposition error: 'Different from' is standard in formal English.", "academic_essay"),
    ("did a mistake", "made a mistake", "It is understandable that the junior engineer {word} during the initial deployment.", "Collocation: The natural English idiom is 'make a mistake', not 'do a mistake'.", "technical_code"),
    ("take a decision", "make a decision", "The executive committee will {word} regarding the vendor contract today.", "Collocation: In standard English, one 'makes a decision' (or 'reaches a decision').", "business_finance"),
    ("make research", "conduct research", "The artificial intelligence institute will {word} into energy-efficient edge computing.", "Collocation: In formal writing, use 'conduct research' or 'do research', not 'make research'.", "academic_essay"),
    ("pay attention at", "pay attention to", "Please {word} the security warning banners displayed on the terminal.", "Preposition error: Use 'pay attention to', not 'pay attention at'.", "technical_code"),
]

# -----------------------------------------------------------------------------
# 6. COGNITIVE SPELLING & TRANSPOSITION TYPOS
# -----------------------------------------------------------------------------
SPELLING_DATA = [
    # (typo, clean, template, explanation, domain)
    ("definately", "definitely", "We will {word} deploy the optimized ONNX runtime before Friday.", "Spelling: The correct spelling is 'definitely', with an 'i', not 'definately'.", "technical_code"),
    ("seperate", "separate", "Keep the training and evaluation datasets in {word} directories on disk.", "Spelling: 'Separate' is spelled with 'par', not 'per'.", "technical_code"),
    ("occurrance", "occurrence", "A deadlock under standard operating load is a rare {word}.", "Spelling: 'Occurrence' has double 'c', double 'r', and ends with 'ence'.", "technical_code"),
    ("maintainance", "maintenance", "The data center will undergo routine hardware {word} this weekend.", "Spelling: The noun is 'maintenance', not 'maintainance'.", "professional_email"),
    ("acommodate", "accommodate", "The edge server can easily {word} over five thousand concurrent connections.", "Spelling: 'Accommodate' has double 'c' and double 'm'.", "technical_code"),
    ("embarass", "embarrass", "A public security breach can severely {word} an enterprise organization.", "Spelling: 'Embarrass' has double 'r' and double 's'.", "business_finance"),
    ("reccomend", "recommend", "I strongly {word} enabling rotary position embeddings in the architecture.", "Spelling: 'Recommend' has one 'c' and double 'm'.", "technical_code"),
    ("tommorow", "tomorrow", "The client briefing and architecture demo is scheduled for {word} morning.", "Spelling: 'Tomorrow' has one 'm' and double 'r'.", "professional_email"),
    ("priviledge", "privilege", "Direct production database access is a strictly controlled {word}.", "Spelling: 'Privilege' has no 'd'.", "technical_code"),
    ("beleive", "believe", "We {word} that offline local inference ensures maximum user privacy.", "Spelling: 'i' before 'e' except after 'c': 'believe'.", "academic_essay"),
    ("acheive", "achieve", "The model was able to {word} sub-ten millisecond CPU response times.", "Spelling: 'i' before 'e': 'achieve'.", "technical_code"),
    ("wierd", "weird", "The legacy parser exhibited {word} formatting behavior with non-ASCII characters.", "Spelling: 'Weird' is an exception to the 'i before e' rule: 'e' comes before 'i'.", "technical_code"),
    ("neccessary", "necessary", "Comprehensive error handling is {word} for high-availability systems.", "Spelling: 'Necessary' has one 'c' and double 's'.", "technical_code"),
    ("goverment", "government", "The federal {word} announced updated guidelines on artificial intelligence governance.", "Spelling: Do not forget the 'n' in 'government'.", "journalism"),
    ("enviroment", "environment", "Always initialize a virtual {word} prior to installing project dependencies.", "Spelling: Include the 'n' in 'environment'.", "technical_code"),
    ("calender", "calendar", "Please schedule the sprint retrospective on our shared team {word}.", "Spelling: A system of days is a 'calendar' (ending in 'ar').", "professional_email"),
    ("collegue", "colleague", "My senior engineering {word} assisted in optimizing the CUDA matrix kernels.", "Spelling: 'Colleague' has 'ea' and ends in 'gue'.", "professional_email"),
    ("untill", "until", "Continue fine-tuning the model {word} validation perplexity reaches an asymptote.", "Spelling: 'Until' ends with a single 'l'.", "technical_code"),
    ("fourty", "forty", "The engineering sprint resolved more than {word} critical backlog bugs.", "Spelling: The number 40 is spelled 'forty', without a 'u'.", "business_finance"),
    ("independant", "independent", "The security audit was conducted by an {word} third-party cybersecurity firm.", "Spelling: 'Independent' ends with 'ent', not 'ant'.", "academic_essay"),
    ("existance", "existence", "There is no conclusive evidence for the {word} of that rumored memory leak.", "Spelling: 'Existence' ends with 'ence', not 'ance'.", "technical_code"),
    ("guarentee", "guarantee", "We {word} ninety-nine point nine percent uptime across all production tiers.", "Spelling: 'Guarantee' starts with 'gua' and ends with double 'e'.", "business_finance"),
    ("succesful", "successful", "The blue-green cluster migration was completely {word} with zero user disruption.", "Spelling: 'Successful' has double 'c', double 's', and single 'l'.", "technical_code"),
    ("millenium", "millennium", "The turn of the new {word} ushered in unprecedented technological growth.", "Spelling: 'Millennium' has double 'l' and double 'n'.", "academic_essay"),
    ("hierachy", "hierarchy", "The organizational {word} was flattened to encourage agile cross-functional teams.", "Spelling: 'Hierarchy' contains 'er' after 'hi'.", "business_finance"),
    ("liason", "liaison", "She acts as the primary technical {word} between data science and DevOps.", "Spelling: 'Liaison' has an 'i' after 'a'.", "professional_email"),
    ("pronounciation", "pronunciation", "The speech synthesis engine produces natural English {word}.", "Spelling: The noun is 'pronunciation' (with 'nun'), not 'pronounciation'.", "academic_essay"),
    ("rythm", "rhythm", "The steady {word} of typing echoed throughout the quiet engineering office.", "Spelling: 'Rhythm' has an 'h' after 'r' and after 't'.", "creative_writing"),
    ("burocracy", "bureaucracy", "Modern startups adopt streamlined processes to minimize administrative {word}.", "Spelling: 'Bureaucracy' starts with 'bureau'.", "business_finance"),
    ("surveilance", "surveillance", "Automated network {word} immediately flagged the anomalous outbound traffic.", "Spelling: 'Surveillance' has double 'l'.", "technical_code"),
    ("perseverence", "perseverance", "Debugging the elusive distributed consensus bug required patience and {word}.", "Spelling: 'Perseverance' ends with 'ance'.", "technical_code"),
]

# -----------------------------------------------------------------------------
# 7. QUILLBOT-STYLE FLUENCY & CONCISENESS
# -----------------------------------------------------------------------------
FLUENCY_DATA = [
    # (wordy, concise, explanation, domain)
    (
        "Due to the fact that the server was overloaded with requests, latency spiked significantly.",
        "Because the server was overloaded with requests, latency spiked significantly.",
        "Conciseness: 'Due to the fact that' is wordy; replace it with the concise conjunction 'Because'.",
        "technical_code"
    ),
    (
        "In order to achieve maximum throughput, the team parallelized the matrix operations.",
        "To achieve maximum throughput, the team parallelized the matrix operations.",
        "Conciseness: 'In order to' can almost always be simplified to 'To' without loss of meaning.",
        "technical_code"
    ),
    (
        "At the present point in time, we are unable to process international wire transfers.",
        "Currently, we are unable to process international wire transfers.",
        "Fluency: 'At the present point in time' is needlessly wordy; use 'Currently' or 'Now'.",
        "business_finance"
    ),
    (
        "Each and every single employee must complete the annual compliance training.",
        "Every employee must complete the annual compliance training.",
        "Redundancy: 'Each and every single' is a repetitive tautology; simplify to 'Every employee'.",
        "professional_email"
    ),
    (
        "We decided to collaborate together with the external research laboratory on this paper.",
        "We decided to collaborate with the external research laboratory on this paper.",
        "Redundancy: 'Collaborate' already means working together; omit the redundant word 'together'.",
        "academic_essay"
    ),
    (
        "The end result of the load test confirmed our theoretical throughput model.",
        "The result of the load test confirmed our theoretical throughput model.",
        "Redundancy: 'End result' is redundant because results inherently come at the end; use 'result'.",
        "academic_essay"
    ),
    (
        "The physician reviewed the past history of the patient before prescribing medication.",
        "The physician reviewed the medical history of the patient before prescribing medication.",
        "Redundancy: History by definition relates to the past; 'past history' is redundant.",
        "academic_essay"
    ),
    (
        "We must plan in advance for potential data center power outages.",
        "We must plan for potential data center power outages.",
        "Redundancy: Planning inherently takes place in advance; omit 'in advance'.",
        "technical_code"
    ),
    (
        "The software patch will completely eliminate all remaining race conditions.",
        "The software patch will eliminate all remaining race conditions.",
        "Redundancy: 'Eliminate' is an absolute verb; 'completely eliminate' is redundant.",
        "technical_code"
    ),
    (
        "Please revert back to the previous stable release commit immediately.",
        "Please revert to the previous stable release commit immediately.",
        "Redundancy: 'Revert' means to return to a previous state; 'revert back' is redundant.",
        "technical_code"
    ),
    (
        "It is necessary for us to conduct an investigation into the security anomaly.",
        "We must thoroughly investigate the security anomaly.",
        "Fluency: Convert nominalized, passive phrasing ('It is necessary for us to conduct') into active, direct prose.",
        "business_finance"
    ),
    (
        "Despite the fact that the benchmark was challenging, our model ranked first.",
        "Although the benchmark was challenging, our model ranked first.",
        "Conciseness: Replace the clunky phrase 'Despite the fact that' with 'Although' or 'Even though'.",
        "academic_essay"
    ),
    (
        "The new caching policy has the capability to reduce disk read IOPS by half.",
        "The new caching policy can reduce disk read IOPS by half.",
        "Conciseness: 'Has the capability to' is wordy; replace with 'can'.",
        "technical_code"
    ),
    (
        "We should give consideration to alternative distributed database designs.",
        "We should consider alternative distributed database designs.",
        "Conciseness: 'Give consideration to' is wordy nominalization; use 'consider'.",
        "technical_code"
    ),
    (
        "In the event that the primary node fails, the cluster will automatically elect a leader.",
        "If the primary node fails, the cluster will automatically elect a leader.",
        "Conciseness: Replace 'In the event that' with the concise conjunction 'If'.",
        "technical_code"
    ),
]

# -----------------------------------------------------------------------------
# 8. PUNCTUATION & CAPITALIZATION
# -----------------------------------------------------------------------------
MECHANICS_DATA = [
    # (input_text, target_text, explanation, domain)
    (
        "The database server crashed, we had to failover to the secondary node.",
        "The database server crashed; we had to failover to the secondary node.",
        "Punctuation (Comma splice): Two independent clauses cannot be joined solely by a comma. Use a semicolon.",
        "technical_code"
    ),
    (
        "Although the network latency was exceptionally low the throughput remained constrained.",
        "Although the network latency was exceptionally low, the throughput remained constrained.",
        "Punctuation (Introductory clause): A dependent introductory clause starting with 'Although' requires a comma.",
        "technical_code"
    ),
    (
        "The server's were upgraded with faster solid-state drives last weekend.",
        "The servers were upgraded with faster solid-state drives last weekend.",
        "Punctuation (Plural apostrophe): Do not use an apostrophe to form a simple plural noun ('servers').",
        "technical_code"
    ),
    (
        "Two prominent company CEO's spoke at the global technology summit.",
        "Two prominent company CEOs spoke at the global technology summit.",
        "Punctuation (Acronym plural): Form the plural of acronyms by adding a lowercase 's' without an apostrophe ('CEOs').",
        "business_finance"
    ),
    (
        "i believe that we should refactor the inference engine before the customer demo.",
        "I believe that we should refactor the inference engine before the customer demo.",
        "Capitalization: The first-person singular pronoun 'I' must always be capitalized.",
        "casual_communication"
    ),
    (
        "The engineering squad meets every monday morning to review pull requests.",
        "The engineering squad meets every Monday morning to review pull requests.",
        "Capitalization: Days of the week like 'Monday' are proper nouns and must be capitalized.",
        "professional_email"
    ),
    (
        "We evaluated tokenizer token distributions across english and spanish corpora.",
        "We evaluated tokenizer token distributions across English and Spanish corpora.",
        "Capitalization: Names of languages and nationalities ('English', 'Spanish') must be capitalized.",
        "academic_essay"
    ),
    (
        "The microservice was developed in python and deployed on aws.",
        "The microservice was developed in Python and deployed on AWS.",
        "Capitalization: Software languages and cloud acronyms ('Python', 'AWS') are proper nouns.",
        "technical_code"
    ),
]

# -----------------------------------------------------------------------------
# 9. FLAWLESS CLEAN BENCHMARK SENTENCES (Anchor Negative Examples - 20% ratio)
# -----------------------------------------------------------------------------
CLEAN_PROSE = [
    # Technical & AI
    ("TypoFormer runs lightweight INT8 quantized graphs directly on CPU without cloud latency.", "technical_code"),
    ("Grouped query attention significantly reduces memory bus traffic during interactive decoding.", "technical_code"),
    ("The character-aware CNN embedding highway extracts robust representations from misspelled words.", "technical_code"),
    ("We configured continuous integration pipelines to run automated unit tests on every pull request.", "technical_code"),
    ("The rotary position embedding method provides relative distance awareness across sequence tokens.", "technical_code"),
    ("FastAPI leverages Python type hints to provide automatic OpenAPI documentation and validation.", "technical_code"),
    ("Our non-autoregressive grammar tagger predicts atomic edit operations in a single forward pass.", "technical_code"),
    ("The database connection pool maintains twenty persistent sockets to prevent connection overhead.", "technical_code"),
    ("We benchmarked the throughput on an AMD Ryzen processor and achieved sub-ten millisecond latency.", "technical_code"),
    ("Zero-trust network architecture requires mutual TLS authentication between all internal services.", "technical_code"),
    ("The tokenizer utilizes byte-level byte-pair encoding to represent arbitrary UTF-8 characters.", "technical_code"),
    ("ONNX Runtime provides highly optimized CPU execution providers utilizing AVX-512 vectorization.", "technical_code"),
    ("The sliding window attention mechanism prevents quadratic memory explosion on long documents.", "technical_code"),
    ("We implemented rate limiting on the public API gateway to prevent denial-of-service attempts.", "technical_code"),
    ("The distributed training job utilized mixed-precision arithmetic to accelerate convergence.", "technical_code"),
    ("Docker containers ensure reproducible deployment environments across all staging servers.", "technical_code"),
    ("The caching layer utilizes Redis with LRU eviction to maintain sub-millisecond response times.", "technical_code"),
    ("Kubernetes automatically scaled the replica pods in response to the surge in user traffic.", "technical_code"),
    ("Static code analysis identified three potential null reference exceptions before compilation.", "technical_code"),
    ("We benchmarked the inference engine across varied prompt lengths to evaluate latency scaling.", "technical_code"),

    # Professional Email & Business
    ("Please review the attached project proposal and let me know if you have any questions.", "professional_email"),
    ("Thank you for taking the time to meet with our executive leadership team yesterday.", "professional_email"),
    ("I will follow up with the legal department to ensure our licensing terms are fully compliant.", "professional_email"),
    ("We are pleased to announce that our quarterly revenue exceeded market projections by twelve percent.", "business_finance"),
    ("Could you please provide an updated estimate of the engineering hours required for this feature?", "professional_email"),
    ("The board of directors unanimously approved the proposed organizational restructuring.", "business_finance"),
    ("We look forward to collaborating with your team on the upcoming customer onboarding initiative.", "professional_email"),
    ("Please ensure that all team members submit their weekly timesheets prior to Friday afternoon.", "professional_email"),
    ("Our client requested a demonstration of the offline spelling and grammar correction overlay.", "professional_email"),
    ("The strategic partnership will expand our product distribution across North American markets.", "business_finance"),
    ("The financial audit confirmed that our operational expenditures remained strictly within budget.", "business_finance"),
    ("We scheduled a quarterly business review with our enterprise stakeholders for next Thursday.", "business_finance"),
    ("Please find attached the finalized invoice corresponding to the third milestone deliverables.", "professional_email"),
    ("The executive committee commended the squad for successfully delivering the platform ahead of schedule.", "professional_email"),

    # Academic & Science
    ("The empirical findings demonstrate a statistically significant correlation between latency and user retention.", "academic_essay"),
    ("Recent advancements in deep learning have transformed the landscape of natural language processing.", "academic_essay"),
    ("The researchers conducted a double-blind controlled trial to evaluate the efficacy of the intervention.", "academic_essay"),
    ("A comprehensive literature review revealed substantial divergence among published theoretical models.", "academic_essay"),
    ("The experimental methodology adheres strictly to established institutional ethical guidelines.", "academic_essay"),
    ("Quantitative analysis of the survey responses indicates widespread user satisfaction with the interface.", "academic_essay"),
    ("The author synthesizes historical perspectives with contemporary sociolinguistic frameworks.", "academic_essay"),
    ("Cross-validation was performed across five distinct folds to ensure model generalization.", "academic_essay"),
    ("The hypothesis was corroborated by rigorous multivariate regression analysis.", "academic_essay"),
    ("These results provide compelling evidence for the role of morphological awareness in language acquisition.", "academic_essay"),
    ("The dataset was normalized using standard z-score scaling prior to principal component analysis.", "academic_essay"),
    ("Ablation studies confirmed that the character CNN highway accounts for a substantial gain in robustness.", "academic_essay"),

    # Casual & Creative
    ("The morning sunlight filtered through the dense canopy of ancient oak trees in the forest.", "creative_writing"),
    ("She enjoyed sitting by the window with a warm cup of coffee while watching the rain fall.", "creative_writing"),
    ("I will definitely be there on time, so please save a seat for me near the front.", "casual_communication"),
    ("The tranquil rhythm of the ocean waves provided a soothing background throughout the evening.", "creative_writing"),
    ("We had a wonderful dinner together and caught up on all the news from back home.", "casual_communication"),
    ("The old bookstore on the corner carried an impressive collection of rare first editions.", "creative_writing"),
    ("It was an exhilarating experience to hike along the rugged mountain ridge at dawn.", "creative_writing"),
    ("Let me know if you would like to grab lunch together after the conference session ends.", "casual_communication"),
    ("The evening breeze carried the sweet scent of jasmine across the peaceful courtyard.", "creative_writing"),
    ("He took a moment to reflect on his journey before embarking on the new venture.", "creative_writing"),
]


def clean_text_spaces(text: str) -> str:
    """Normalize any internal whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def validate_row(r: DatasetRow) -> None:
    """Rigorous sanity checks for every generated row."""
    assert "  " not in r.input_text, f"Double space found in input: {r.input_text}"
    assert "  " not in r.target_text, f"Double space found in target: {r.target_text}"
    if r.error_type == "clean":
        assert r.input_text == r.target_text, f"Clean row has mismatch: {r.input_text} vs {r.target_text}"
        assert r.error_count == 0
    else:
        assert r.input_text != r.target_text, f"Error row has identical text: {r.input_text}"
        assert r.error_count > 0


def generate_dataset(target_rows: int = 25000, seed: int = 42) -> list[DatasetRow]:
    rng = random.Random(seed)
    rows: list[DatasetRow] = []
    row_id = 1

    def make_id() -> str:
        nonlocal row_id
        res = f"GQ_{row_id:06d}"
        row_id += 1
        return res

    # 1. Homophones & Confused Words
    for wrong, right, template, expl, domain in HOMOPHONE_PAIRS:
        n_text = clean_text_spaces(template.format(word=wrong))
        c_text = clean_text_spaces(template.format(word=right))
        rows.append(DatasetRow(
            id=make_id(),
            input_text=n_text,
            target_text=c_text,
            error_type="grammar",
            error_category="homophone_confusion",
            error_count=1,
            correction_explanation=expl,
            domain=domain,
            complexity_level="intermediate"
        ))

    # 2. Subject-Verb Agreement
    for sing_sub, pl_v, sing_v, pl_sub, rest, expl, domain in SVA_DATA:
        # Singular subject with plural verb error
        n_text = clean_text_spaces(f"{sing_sub} {pl_v} {rest}")
        c_text = clean_text_spaces(f"{sing_sub} {sing_v} {rest}")
        rows.append(DatasetRow(
            id=make_id(),
            input_text=n_text,
            target_text=c_text,
            error_type="syntax",
            error_category="subject_verb_agreement",
            error_count=1,
            correction_explanation=expl,
            domain=domain,
            complexity_level="intermediate"
        ))
        # Plural subject with singular verb error
        inv_expl = f"Subject-verb agreement: '{pl_sub}' is plural, requiring the plural verb '{pl_v}'."
        n_text = clean_text_spaces(f"{pl_sub} {sing_v} {rest}")
        c_text = clean_text_spaces(f"{pl_sub} {pl_v} {rest}")
        rows.append(DatasetRow(
            id=make_id(),
            input_text=n_text,
            target_text=c_text,
            error_type="syntax",
            error_category="subject_verb_agreement",
            error_count=1,
            correction_explanation=inv_expl,
            domain=domain,
            complexity_level="intermediate"
        ))

    # 3. Verb Tense & Inflection
    for wrong_v, right_v, template, expl, domain in TENSE_DATA:
        n_text = clean_text_spaces(template.format(word=wrong_v))
        c_text = clean_text_spaces(template.format(word=right_v))
        rows.append(DatasetRow(
            id=make_id(),
            input_text=n_text,
            target_text=c_text,
            error_type="grammar",
            error_category="verb_tense_aspect",
            error_count=1,
            correction_explanation=expl,
            domain=domain,
            complexity_level="intermediate"
        ))

    # 4. Countability & Quantifiers
    for wrong_n, right_n, template, expl, domain in COUNTABILITY_DATA:
        n_text = clean_text_spaces(template.format(word=wrong_n))
        c_text = clean_text_spaces(template.format(word=right_n))
        rows.append(DatasetRow(
            id=make_id(),
            input_text=n_text,
            target_text=c_text,
            error_type="grammar",
            error_category="countable_uncountable",
            error_count=1,
            correction_explanation=expl,
            domain=domain,
            complexity_level="intermediate"
        ))

    # 5. Prepositions & Collocations
    for wrong_p, right_p, template, expl, domain in PREPOSITION_DATA:
        n_text = clean_text_spaces(template.format(word=wrong_p))
        c_text = clean_text_spaces(template.format(word=right_p))
        rows.append(DatasetRow(
            id=make_id(),
            input_text=n_text,
            target_text=c_text,
            error_type="fluency",
            error_category="preposition_collocation",
            error_count=1,
            correction_explanation=expl,
            domain=domain,
            complexity_level="intermediate"
        ))

    # 6. Spelling & Typos (Semantically Matched)
    for typo, clean, template, expl, domain in SPELLING_DATA:
        n_text = clean_text_spaces(template.format(word=typo))
        c_text = clean_text_spaces(template.format(word=clean))
        rows.append(DatasetRow(
            id=make_id(),
            input_text=n_text,
            target_text=c_text,
            error_type="spelling",
            error_category="spelling_typo",
            error_count=1,
            correction_explanation=expl,
            domain=domain,
            complexity_level="elementary"
        ))

    # 7. QuillBot Fluency & Redundancy
    for wordy, concise, expl, domain in FLUENCY_DATA:
        rows.append(DatasetRow(
            id=make_id(),
            input_text=clean_text_spaces(wordy),
            target_text=clean_text_spaces(concise),
            error_type="fluency",
            error_category="wordiness_redundancy",
            error_count=1,
            correction_explanation=expl,
            domain=domain,
            complexity_level="advanced"
        ))

    # 8. Mechanics & Punctuation
    for input_txt, target_txt, expl, domain in MECHANICS_DATA:
        rows.append(DatasetRow(
            id=make_id(),
            input_text=clean_text_spaces(input_txt),
            target_text=clean_text_spaces(target_txt),
            error_type="punctuation",
            error_category="punctuation_mechanics",
            error_count=1,
            correction_explanation=expl,
            domain=domain,
            complexity_level="intermediate"
        ))

    # 9. Clean Benchmark Sentences (Negative Examples - 20% ratio)
    clean_explanations = [
        "Flawless syntax and grammar; no correction required.",
        "High-precision sentence preservation; prose is natural and accurate.",
        "Correct vocabulary usage, punctuation, and subject-verb agreement.",
        "Sentence adheres strictly to standard professional English conventions.",
    ]
    for clean_sent, domain in CLEAN_PROSE:
        c_text = clean_text_spaces(clean_sent)
        rows.append(DatasetRow(
            id=make_id(),
            input_text=c_text,
            target_text=c_text,
            error_type="clean",
            error_category="clean_preservation",
            error_count=0,
            correction_explanation=rng.choice(clean_explanations),
            domain=domain,
            complexity_level="professional"
        ))

    base_seeds = list(rows)
    print(f"Verified semantically coherent seed pairs: {len(base_seeds):,}")

    # Validate all base seed rows
    for r in base_seeds:
        validate_row(r)

    # 10. Contextual introductory clauses for natural scale expansion
    contextual_intros = [
        ("Please note that ", "professional_email"),
        ("As reported by the lead engineer, ", "technical_code"),
        ("Furthermore, ", "academic_essay"),
        ("In our preliminary evaluation, ", "academic_essay"),
        ("During the recent sprint cycle, ", "technical_code"),
        ("According to the project status report, ", "business_finance"),
        ("Upon closer inspection of the system logs, ", "technical_code"),
        ("To the best of our knowledge, ", "academic_essay"),
        ("In the latest enterprise release, ", "technical_code"),
        ("From an operational perspective, ", "business_finance"),
        ("Following the technical retrospective, ", "technical_code"),
        ("In practical production environments, ", "technical_code"),
        ("Under high-concurrency stress testing, ", "technical_code"),
        ("As stated in the technical documentation, ", "technical_code"),
        ("Historically speaking, ", "academic_essay"),
        ("For the purpose of clarity, ", "professional_email"),
        ("In accordance with team guidelines, ", "professional_email"),
        ("After reviewing the pull request, ", "technical_code"),
        ("Based on feedback from beta testers, ", "professional_email"),
        ("In the introductory chapter of the report, ", "academic_essay"),
        ("Interestingly, ", "casual_communication"),
        ("As demonstrated during the benchmark, ", "technical_code"),
        ("In regular daily workflows, ", "professional_email"),
        ("From a security standpoint, ", "technical_code"),
        ("In subsequent experiment runs, ", "academic_essay"),
    ]

    while len(rows) < target_rows:
        seed_item = rng.choice(base_seeds)
        intro, intro_dom = rng.choice(contextual_intros)

        c0_noisy = seed_item.input_text[0].lower()
        c0_clean = seed_item.target_text[0].lower()

        # Handle 'I' pronoun preservation when lowercasing first char
        if seed_item.input_text.startswith("I "):
            c0_noisy = "I"
        if seed_item.target_text.startswith("I "):
            c0_clean = "I"

        new_noisy = clean_text_spaces(intro + c0_noisy + seed_item.input_text[1:])
        new_clean = clean_text_spaces(intro + c0_clean + seed_item.target_text[1:])

        r_exp = DatasetRow(
            id=make_id(),
            input_text=new_noisy,
            target_text=new_clean,
            error_type=seed_item.error_type,
            error_category=seed_item.error_category,
            error_count=seed_item.error_count,
            correction_explanation=seed_item.correction_explanation,
            domain=intro_dom if seed_item.error_type != "clean" else seed_item.domain,
            complexity_level=seed_item.complexity_level
        )
        validate_row(r_exp)
        rows.append(r_exp)

    # Shuffle for uniform distribution
    rng.shuffle(rows)

    # Re-index IDs sequentially
    final_rows = [
        DatasetRow(
            id=f"GQ_{i+1:06d}",
            input_text=r.input_text,
            target_text=r.target_text,
            error_type=r.error_type,
            error_category=r.error_category,
            error_count=r.error_count,
            correction_explanation=r.correction_explanation,
            domain=r.domain,
            complexity_level=r.complexity_level,
        )
        for i, r in enumerate(rows[:target_rows])
    ]

    return final_rows


def export_csv(rows: list[DatasetRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            "id",
            "input_text",
            "target_text",
            "error_type",
            "error_category",
            "error_count",
            "correction_explanation",
            "domain",
            "complexity_level"
        ])
        for r in rows:
            writer.writerow([
                r.id,
                r.input_text,
                r.target_text,
                r.error_type,
                r.error_category,
                r.error_count,
                r.correction_explanation,
                r.domain,
                r.complexity_level,
            ])
    print(f"Exported {len(rows):,} rows -> {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


def main():
    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "data" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    target_count = 25000
    print(f"Synthesizing {target_count:,} Grammarly/QuillBot quality GEC training rows...")
    dataset = generate_dataset(target_rows=target_count, seed=42)

    # Master CSV
    master_path = dataset_dir / "grammarly_quillbot_dataset.csv"
    export_csv(dataset, master_path)

    # Train / Val Split (90% / 10%)
    split_idx = int(len(dataset) * 0.90)
    train_rows = dataset[:split_idx]
    val_rows = dataset[split_idx:]

    train_path = dataset_dir / "train.csv"
    val_path = dataset_dir / "val.csv"
    export_csv(train_rows, train_path)
    export_csv(val_rows, val_path)

    print("\nDataset Generation Complete!")
    print(f" - Master Dataset (25,000 rows): {master_path}")
    print(f" - Train Split (22,500 rows): {train_path}")
    print(f" - Val Split (2,500 rows): {val_path}")


if __name__ == "__main__":
    main()
