"""
High-Precision Wine-Grade Dataset Generator for TypoFormer.
Guarantees 100% natural, semantically coherent English sentences across:
1. Technical & Engineering (Architecture, PRs, Git, APIs, DevOps, Cloud, DB, Concurrency, Latency)
2. Client Handling & Professional Communication (Milestones, Deliverables, Inquiries, SLAs, Agreements)
3. Workplace & Team Collaboration (Slack, Teams, Quick Huddles, Daily Standups, Casual Syncs)

Grammatically Slotted:
- Nouns only into noun slots
- Verbs only into verb slots
- Adjectives only into adjective slots
- Adverbs only into adverb slots
Zero word-salad. Zero ungrammatical template cross-products.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import NamedTuple


class WineRow(NamedTuple):
    id: str
    input_text: str
    target_text: str
    error_type: str
    error_category: str
    error_count: int
    correction_explanation: str
    domain: str
    complexity_level: str


# =============================================================================
# 1. TECHNICAL HOMOPHONES & CONFUSED WORDS (NATURAL PAIRS)
# =============================================================================
TECH_HOMOPHONES = [
    # affect / effect
    ("effect", "affect", "High network jitter will directly {word} the streaming pipeline throughput.", "Use 'affect' as a verb meaning to influence.", "engineering"),
    ("affect", "effect", "The database index optimization had an immediate positive {word} on query latency.", "Use 'effect' as a noun meaning result or outcome.", "engineering"),
    ("effected", "affected", "Several microservices were adversely {word} by the unexpected outage.", "Use 'affected' as the past participle verb.", "engineering"),
    ("effects", "affects", "This memory leak negatively {word} our overall server availability.", "Use 'affects' as third-person singular verb.", "engineering"),
    ("affective", "effective", "Rate limiting is an {word} countermeasure against distributed denial of service attacks.", "Use 'effective' to mean producing desired results.", "engineering"),

    # their / there / they're
    ("their", "there", "Please verify that {word} are no remaining lint warnings in the build log.", "Use 'there' to indicate existence or location.", "engineering"),
    ("there", "their", "The infrastructure team pushed {word} updated Helm charts to staging.", "Use 'their' for possession.", "engineering"),
    ("they're", "their", "Developers must tag {word} pull requests with appropriate reviewers.", "Use 'their' for possession; 'they\\'re' means 'they are'.", "engineering"),
    ("their", "they're", "The platform engineers confirmed {word} preparing the hotfix for production.", "Use 'they\\'re' as a contraction of 'they are'.", "engineering"),
    ("there", "they're", "We checked with DevOps and {word} deploying the container image now.", "Use 'they\\'re' for 'they are'.", "engineering"),

    # its / it's
    ("it's", "its", "The transformer backbone reached {word} lowest cross-entropy loss at step 2000.", "Use 'its' as the possessive pronoun.", "engineering"),
    ("its", "it's", "Before running the migration script, {word} crucial to take a snapshot of the volume.", "Use 'it\\'s' for 'it is'.", "engineering"),
    ("it's", "its", "The garbage collector released {word} allocated memory back to the operating system.", "Use 'its' as the possessive pronoun.", "engineering"),
    ("its", "it's", "When a worker crashes, {word} automatically restarted by the orchestrator.", "Use 'it\\'s' as contraction for 'it is'.", "engineering"),
    ("it's", "its", "The microservice failed to renew {word} authentication token before expiring.", "Use 'its' as the possessive pronoun.", "engineering"),

    # principal / principle
    ("principal", "principle", "Our team adheres strictly to the architectural {word} of single responsibility.", "Use 'principle' for a fundamental law or rule.", "engineering"),
    ("principle", "principal", "The {word} software engineer reviewed the proposed distributed cache design.", "Use 'principal' for a lead title or primary authority.", "engineering"),
    ("principal", "principle", "Decoupling services is a core design {word} in cloud-native computing.", "Use 'principle' for design rule.", "engineering"),

    # ensure / insure / assure
    ("insure", "ensure", "Please {word} that all unit and integration tests pass before requesting review.", "Use 'ensure' to mean guarantee or make certain.", "engineering"),
    ("ensure", "assure", "I can {word} the team that the API backward compatibility is preserved.", "Use 'assure' to remove doubt or give confidence to people.", "engineering"),
    ("insure", "ensure", "We configured health check probes to {word} high service availability.", "Use 'ensure' to guarantee an outcome.", "engineering"),

    # loose / lose
    ("loose", "lose", "We cannot afford to {word} buffered messages during a broker restart.", "Use 'lose' to mean misplace or suffer loss.", "engineering"),
    ("lose", "loose", "A {word} configuration parameter caused the worker threads to drift.", "Use 'loose' to mean unfastened or unbound.", "engineering"),
    ("loosing", "losing", "The load balancer is currently {word} connection packets due to buffer overrun.", "Use 'losing' from verb 'lose'.", "engineering"),

    # discrete / discreet
    ("discreet", "discrete", "The event bus publishes messages into {word} topic partitions.", "Use 'discrete' to mean distinct, detached, or separate.", "engineering"),
    ("discrete", "discreet", "Please remain {word} regarding the upcoming acquisition until the public announcement.", "Use 'discreet' to mean tactful and circumspect.", "engineering"),

    # accept / except
    ("except", "accept", "The REST endpoint will only {word} valid JSON payloads with an HMAC header.", "Use 'accept' to mean receive or approve.", "engineering"),
    ("accept", "except", "All unit tests succeeded {word} for the legacy LDAP synchronization mock.", "Use 'except' to mean excluding.", "engineering"),

    # site / cite / sight
    ("sight", "site", "The monitoring probe detected that the staging {word} is returning 502 bad gateway.", "Use 'site' for a website or physical location.", "engineering"),
    ("site", "cite", "Make sure to {word} the relevant RFC specifications in your architecture document.", "Use 'cite' to mean reference or quote an authority.", "engineering"),

    # precede / proceed
    ("precede", "proceed", "Once the CI check passes, you can {word} with merging the pull request.", "Use 'proceed' to mean advance or continue forward.", "engineering"),
    ("proceed", "precede", "Thorough schema validation must always {word} database persistence.", "Use 'precede' to mean come before in sequence.", "engineering"),

    # advice / advise
    ("advice", "advise", "The staff database administrator will {word} the squad on partition keys.", "Use 'advise' as the verb.", "engineering"),
    ("advise", "advice", "We took the architect's {word} and decoupled the reporting service.", "Use 'advice' as the noun.", "engineering"),

    # lead / led
    ("lead", "led", "Fahim {word} the performance optimization effort that halved our inference latency.", "Use 'led' as the past tense of 'lead'.", "engineering"),
    ("led", "lead", "Who will {word} the incident retrospective for yesterday's database timeout?", "Use 'lead' as the present tense verb.", "engineering"),

    # cache / cash
    ("cash", "cache", "We invalidated the Redis {word} to force a fresh reload of user session data.", "Use 'cache' for computer memory store.", "engineering"),
    ("cached", "cashed", "The query results are {word} in memory for sixty seconds.", "Use 'cached' for saved in cache memory.", "engineering"),

    # queue / cue
    ("cue", "queue", "The background worker pulls jobs from the message {word} sequentially.", "Use 'queue' for an ordered line of items/tasks.", "engineering"),
    ("que", "queue", "Please check if any tasks are stalled in the execution {word}.", "Use 'queue' for a task sequence.", "engineering"),

    # deprecate / depreciate
    ("depreciate", "deprecate", "We plan to {word} version one of the authentication API in the next release.", "Use 'deprecate' to mark software obsolete.", "engineering"),
    ("depreciated", "deprecated", "The old hashing function was {word} in favor of Argon2id.", "Use 'deprecated' for software obsolescence.", "engineering"),
]

# =============================================================================
# 2. CLIENT HANDLING & BUSINESS HOMOPHONES (NATURAL PAIRS)
# =============================================================================
CLIENT_HOMOPHONES = [
    ("compliment", "complement", "Our customized reporting suite will perfectly {word} your existing analytics stack.", "Use 'complement' to mean complete or enhance harmoniously.", "client_handling"),
    ("complement", "compliment", "The client paid our engineering lead a generous {word} regarding the swift turnaround.", "Use 'compliment' for praise or admiration.", "client_handling"),
    ("ensure", "assure", "I want to {word} you that our tier-one support team is actively addressing the ticket.", "Use 'assure' to reassure a person or client.", "client_handling"),
    ("assure", "ensure", "We have added automated regression tests to {word} that this issue does not recur.", "Use 'ensure' to make certain of an event or condition.", "client_handling"),
    ("insured", "ensured", "Our team {word} that all deliverables adhered to the compliance framework.", "Use 'ensured' to mean made certain.", "client_handling"),
    ("there", "their", "We reviewed the client's specifications and confirmed {word} primary objectives.", "Use 'their' for third-person possession.", "client_handling"),
    ("their", "there", "Please let us know if {word} are any specific requirements you would like to add.", "Use 'there' to indicate existence or presence.", "client_handling"),
    ("your", "you're", "We will notify you immediately once {word} ready to schedule the live demo.", "Use 'you\\'re' when you mean 'you are'.", "client_handling"),
    ("you're", "your", "Thank you for taking the time to share {word} detailed evaluation feedback.", "Use 'your' as the possessive pronoun.", "client_handling"),
    ("your", "you're", "Please let us know if {word} satisfied with the proposed milestone schedule.", "Use 'you\\'re' for 'you are'.", "client_handling"),
    ("except", "accept", "We are pleased to {word} the revised scope agreement and initiate work.", "Use 'accept' to mean agree to or take.", "client_handling"),
    ("accept", "except", "All project milestones have been delivered on schedule {word} for the analytics export.", "Use 'except' to mean excluding.", "client_handling"),
    ("illicit", "elicit", "The discovery workshop was organized to {word} key stakeholder requirements.", "Use 'elicit' to mean draw forth or evoke.", "client_handling"),
    ("stationary", "stationery", "Our legal department will forward the contract on official corporate {word}.", "Use 'stationery' for writing materials.", "client_handling"),
    ("advice", "advise", "Could you please {word} us on your team's availability for next week's sprint review?", "Use 'advise' as the verb form.", "client_handling"),
    ("advise", "advice", "We appreciate your strategic {word} on how to prioritize the backlog features.", "Use 'advice' as the noun.", "client_handling"),
    ("perspective", "prospective", "We prepared an executive deck tailored specifically for {word} enterprise clients.", "Use 'prospective' to mean expected or potential.", "client_handling"),
    ("prospective", "perspective", "From an operational {word}, this automation eliminates three hours of manual data entry.", "Use 'perspective' for point of view.", "client_handling"),
    ("then", "than", "Our service response time is currently faster {word} specified in the service level agreement.", "Use 'than' for comparative expressions.", "client_handling"),
    ("than", "then", "We will complete the final security audit and {word} hand over the credentials.", "Use 'then' to indicate subsequent action in time.", "client_handling"),
    ("corroborate", "collaborate", "We look forward to the opportunity to {word} closely with your engineering group.", "Use 'collaborate' to mean work jointly together.", "client_handling"),
]

# =============================================================================
# 3. WORKPLACE CHITCHAT & COLLABORATION HOMOPHONES
# =============================================================================
CHITCHAT_HOMOPHONES = [
    ("their", "there", "Hey, are you {word}? Quick question about the Figma wireframes.", "Use 'there' for location or presence.", "workplace_chitchat"),
    ("your", "you're", "Let me know when {word} free for a quick five-minute sync.", "Use 'you\\'re' as a contraction of 'you are'.", "workplace_chitchat"),
    ("you're", "your", "Thanks so much for {word} help on resolving that tricky deadlock earlier!", "Use 'your' for possession.", "workplace_chitchat"),
    ("its", "it's", "No rush at all, {word} totally fine if we look at this after lunch.", "Use 'it\\'s' for 'it is'.", "workplace_chitchat"),
    ("to", "too", "That sounds good to me {word}; I'll see you in the meeting room.", "Use 'too' to mean 'also' or 'as well'.", "workplace_chitchat"),
    ("then", "than", "I'd rather finish this unit test today {word} leave it for Monday morning.", "Use 'than' for comparisons.", "workplace_chitchat"),
    ("weather", "whether", "I am not sure {word} Fahim has already reviewed the draft PR.", "Use 'whether' to introduce choices or alternatives.", "workplace_chitchat"),
    ("break", "brake", "I am going to grab a quick coffee {word} and will be right back.", "Use 'break' for a pause or intermission.", "workplace_chitchat"),
    ("whose", "who's", "Do you know {word} leading the sprint retro this afternoon?", "Use 'who\\'s' for 'who is'.", "workplace_chitchat"),
    ("who's", "whose", "Hey, {word} branch is currently deployed to the development sandbox?", "Use 'whose' to indicate possession.", "workplace_chitchat"),
    ("hear", "here", "Can everyone {word} me okay, or is there an echo on my microphone?", "Use 'hear' for perceiving sound.", "workplace_chitchat"),
    ("by", "bye", "I'm heading out for the day; {word} everyone, see you tomorrow!", "Use 'bye' for farewell.", "workplace_chitchat"),
]

# =============================================================================
# 4. TAILORED REALISTIC SPELLING SLIPS (NO GIBBERISH / NO MISMATCHED POS)
# Each entry: (misspelling, correct_word, natural_sentence_template, domain)
# =============================================================================
NATURAL_SPELLING_PAIRS = [
    # Engineering terms
    ("asyncronous", "asynchronous", "We converted the email notification dispatch to an {word} Celery task.", "engineering"),
    ("asychronous", "asynchronous", "The background message processing is fully {word} to avoid blocking the API.", "engineering"),
    ("databse", "database", "Make sure to back up the production {word} before applying the schema migration.", "engineering"),
    ("databaes", "database", "The read replica {word} experienced brief replication lag during peak hours.", "engineering"),
    ("architechture", "architecture", "We finalized the microservice {word} of the new distributed caching layer.", "engineering"),
    ("architecure", "architecture", "The decoupled cloud {word} enables horizontal scaling of all services.", "engineering"),
    ("microsevice", "microservice", "Each {word} communicates asynchronously over Apache Kafka event streams.", "engineering"),
    ("microserivce", "microservice", "The authentication {word} validates incoming OAuth tokens before routing.", "engineering"),
    ("implmentation", "implementation", "The {word} of the character convolution layer runs in under ten milliseconds.", "engineering"),
    ("implementaion", "implementation", "Please review the revised interface {word} in the latest pull request.", "engineering"),
    ("serilization", "serialization", "We optimized JSON {word} to reduce HTTP response serialization overhead.", "engineering"),
    ("configration", "configuration", "Please verify the database connection pool {word} in your local environment.", "engineering"),
    ("configuraiton", "configuration", "The Kubernetes deployment {word} specifies memory and CPU limits.", "engineering"),
    ("enviroment", "environment", "The staging {word} mirrors our production cluster topology and settings.", "engineering"),
    ("environemnt", "environment", "All unit tests pass cleanly in the local development {word}.", "engineering"),
    ("dependancy", "dependency", "We removed the unused third-party {word} to minimize bundle size.", "engineering"),
    ("deprecatd", "deprecated", "The legacy v1 REST endpoint will be {word} in the upcoming platform release.", "engineering"),
    ("syncronization", "synchronization", "Thread {word} is handled via lock-free atomic primitives to eliminate contention.", "engineering"),
    ("middlewear", "middleware", "The rate-limiting {word} rejects requests that exceed sixty calls per minute.", "engineering"),
    ("authenticaion", "authentication", "All public endpoints require HMAC {word} headers for data integrity.", "engineering"),
    ("authntication", "authentication", "The OAuth2 {word} flow redirects the user to the single sign-on portal.", "engineering"),
    ("authorizaton", "authorization", "Role-based {word} checks ensure regular users cannot access administrative views.", "engineering"),
    ("idempodent", "idempotent", "Payment webhook handlers must be strictly {word} to prevent duplicate charges.", "engineering"),
    ("througput", "throughput", "The cluster achieves high {word} by partitioning Kafka messages across nodes.", "engineering"),
    ("latancy", "latency", "Our INT8 quantized model reduced p99 inference {word} to under twenty milliseconds.", "engineering"),
    ("latnecy", "latency", "Network {word} between data centers remained well within acceptable thresholds.", "engineering"),
    ("concurrancy", "concurrency", "The Go backend handles high {word} using lightweight goroutines and channels.", "engineering"),
    ("paramater", "parameter", "Make sure the learning rate {word} is tuned appropriately during warmup.", "engineering"),
    ("compatable", "compatible", "Ensure that the new protobuf schema is backward {word} with legacy clients.", "engineering"),
    ("persistant", "persistent", "We configured {word} volume storage for the stateful database pods.", "engineering"),
    ("occured", "occurred", "A transient deadlock {word} in the worker pool during overnight stress testing.", "engineering"),
    ("seperate", "separate", "We refactored the monolith into {word} independently deployable services.", "engineering"),
    ("maintainance", "maintenance", "We scheduled routine server {word} for Saturday night during off-peak hours.", "engineering"),
    ("privelege", "privilege", "The database user was granted read-only {word} on the analytics reporting tables.", "engineering"),
    ("intialize", "initialize", "The application failed to {word} the Redis connection cache on startup.", "engineering"),
    ("repositry", "repository", "Please commit and push your feature branch to the remote Git {word}.", "engineering"),
    ("exepction", "exception", "An unhandled null reference {word} caused the worker thread to terminate.", "engineering"),
    ("transation", "transaction", "All ledger updates are executed inside an ACID {word} block.", "engineering"),
    ("benchamrk", "benchmark", "We ran a performance {word} comparing the FP32 model with the INT8 ONNX graph.", "engineering"),
    ("partiton", "partition", "Kafka consumer lag increased after rebalancing the topic {word} distribution.", "engineering"),
    ("conatiner", "container", "The Docker {word} was built using multi-stage builds to minimize image size.", "engineering"),
    ("alogrithm", "algorithm", "We implemented a custom string matching {word} for fast keystroke correction.", "engineering"),

    # Client & Business terms
    ("delieverable", "deliverable", "We are on track to hand over the first key {word} by close of business Friday.", "client_handling"),
    ("deliverible", "deliverable", "All sprint {word} items have been reviewed and accepted by the client.", "client_handling"),
    ("milistone", "milestone", "We successfully reached the second project {word} ahead of the target deadline.", "client_handling"),
    ("milstone", "milestone", "The revised roadmap outlines clear {word} dates for the enterprise rollout.", "client_handling"),
    ("reponsive", "responsive", "Our support engineering squad remains highly {word} to customer inquiries.", "client_handling"),
    ("scheudle", "schedule", "The team adjusted our sprint {word} to align with your product launch date.", "client_handling"),
    ("calender", "calendar", "I sent a meeting invite to your {word} for the upcoming quarterly business review.", "client_handling"),
    ("recieved", "received", "Thank you for your prompt response; we have {word} your signed agreement.", "client_handling"),
    ("definitly", "definitely", "We will {word} incorporate your feedback into the upcoming sprint deliverables.", "client_handling"),
    ("accomodate", "accommodate", "We are happy to {word} your team's request for additional demo sessions.", "client_handling"),
    ("committment", "commitment", "Our team maintains a firm {word} to delivering high-quality software solutions.", "client_handling"),
    ("recommedation", "recommendation", "Based on our performance audit, our primary {word} is to enable Redis caching.", "client_handling"),
    ("garantee", "guarantee", "Our service agreement provides a ninety-nine point nine percent uptime {word}.", "client_handling"),
    ("neccessary", "necessary", "We will make all {word} adjustments to ensure seamless platform onboarding.", "client_handling"),
    ("oppurtunity", "opportunity", "Thank you for the {word} to partner with your enterprise organization.", "client_handling"),
    ("transparant", "transparent", "We are committed to maintaining a completely {word} reporting workflow throughout.", "client_handling"),
    ("succesful", "successful", "Congratulations on the {word} launch of your new customer-facing portal!", "client_handling"),
    ("clarifiation", "clarification", "Could you provide further {word} on the custom export formatting requirements?", "client_handling"),
    ("concensus", "consensus", "The executive committee reached a unanimous {word} on the proposed roadmap.", "client_handling"),
    ("confidencial", "confidential", "All shared customer analytics data is treated as strictly {word} information.", "client_handling"),
    ("satisfation", "satisfaction", "Client {word} and reliability are the cornerstone metrics of our partnership.", "client_handling"),

    # Casual & Workplace terms
    ("tommorow", "tomorrow", "Hey everyone, let's sync up {word} morning right after the daily standup.", "workplace_chitchat"),
    ("availible", "available", "Are you {word} for a quick coffee catch-up later this afternoon?", "workplace_chitchat"),
    ("appriciate", "appreciate", "I really {word} you stepping in to help with the customer escalations yesterday.", "workplace_chitchat"),
    ("collegue", "colleague", "My team {word} mentioned that you had some great insights on this topic.", "workplace_chitchat"),
    ("immediatly", "immediately", "I will look into that pull request review {word} after this sprint meeting.", "workplace_chitchat"),
    ("queston", "question", "Quick {word} for the group: has anyone tested the latest staging deployment?", "workplace_chitchat"),
    ("probly", "probably", "The CI pipeline timeout is {word} caused by a slow Docker layer download.", "workplace_chitchat"),
    ("wekend", "weekend", "Have a wonderful {word} everyone, and see you all on Monday morning!", "workplace_chitchat"),
    ("apologise", "apologize", "I {word} for dropping off the call; my home Wi-Fi briefly disconnected.", "workplace_chitchat"),
    ("meting", "meeting", "Let's catch up right after the engineering all-hands {word} today.", "workplace_chitchat"),
    ("awsome", "awesome", "That was an {word} presentation on distributed consensus algorithms!", "workplace_chitchat"),
]

# =============================================================================
# 5. PUNCTUATION & CAPITALIZATION SCENARIOS (60+ NATURAL SCENARIOS)
# =============================================================================
PUNCTUATION_SCENARIOS = [
    # Sentence-initial capitalization
    ("we should review the memory footprint before running the benchmark.",
     "We should review the memory footprint before running the benchmark.",
     "Capitalize the first word of a sentence.", "engineering"),
    ("the client confirmed that the demo exceeded their expectations.",
     "The client confirmed that the demo exceeded their expectations.",
     "Capitalize the first word of a sentence.", "client_handling"),
    ("hey Fahim, thanks for jumping in to help debug the redis connection.",
     "Hey Fahim, thanks for jumping in to help debug the redis connection.",
     "Capitalize the first word of a sentence.", "workplace_chitchat"),
    ("are you available for a quick five-minute call to align on priorities?",
     "Are you available for a quick five-minute call to align on priorities?",
     "Capitalize the first word of a sentence.", "workplace_chitchat"),
    ("our team is preparing the sprint retrospective notes for Friday.",
     "Our team is preparing the sprint retrospective notes for Friday.",
     "Capitalize the first word of a sentence.", "workplace_chitchat"),

    # Introductory clause commas
    ("In addition we observed a significant decrease in p99 database latency.",
     "In addition, we observed a significant decrease in p99 database latency.",
     "Use a comma after introductory phrases.", "engineering"),
    ("As discussed during our call yesterday we updated the project timeline.",
     "As discussed during our call yesterday, we updated the project timeline.",
     "Use a comma after introductory dependent clauses.", "client_handling"),
    ("To optimize memory allocation the engineers enabled jemalloc in Docker.",
     "To optimize memory allocation, the engineers enabled jemalloc in Docker.",
     "Use a comma after introductory infinitive phrases.", "engineering"),
    ("Following your feedback we made the necessary adjustments to the API schema.",
     "Following your feedback, we made the necessary adjustments to the API schema.",
     "Use a comma after introductory participial phrases.", "client_handling"),
    ("Before you head out today could you please approve the staging deployment?",
     "Before you head out today, could you please approve the staging deployment?",
     "Use a comma after introductory clauses.", "workplace_chitchat"),
    ("In the meantime our team will continue monitoring the production logs.",
     "In the meantime, our team will continue monitoring the production logs.",
     "Use a comma after transitional phrases.", "client_handling"),
    ("Furthermore the new architecture reduces server provisioning overhead by forty percent.",
     "Furthermore, the new architecture reduces server provisioning overhead by forty percent.",
     "Use a comma after conjunctive adverb 'Furthermore'.", "engineering"),
    ("Consequently the client approved the extended timeline without reservations.",
     "Consequently, the client approved the extended timeline without reservations.",
     "Use a comma after 'Consequently'.", "client_handling"),

    # Direct address commas
    ("Hey Fahim do you have a moment to inspect this trace?",
     "Hey Fahim, do you have a moment to inspect this trace?",
     "Use a comma when directly addressing someone.", "workplace_chitchat"),
    ("Thank you Alex for providing the updated acceptance criteria.",
     "Thank you, Alex, for providing the updated acceptance criteria.",
     "Set off names in direct address with commas.", "client_handling"),
    ("Good morning team let's ensure our tickets are updated before standup.",
     "Good morning team, let's ensure our tickets are updated before standup.",
     "Use a comma after greeting before the clause.", "workplace_chitchat"),

    # Oxford commas in series / lists
    ("The pull request updates authentication serialization and database indexing.",
     "The pull request updates authentication, serialization, and database indexing.",
     "Use commas to separate items in a series.", "engineering"),
    ("Our proposal includes software design system integration and ongoing support.",
     "Our proposal includes software design, system integration, and ongoing support.",
     "Use commas to separate items in a series.", "client_handling"),
    ("I brought coffee donuts and fruit for the morning planning session.",
     "I brought coffee, donuts, and fruit for the morning planning session.",
     "Use commas in a list of items.", "workplace_chitchat"),

    # Compound sentence coordinating conjunction commas
    ("The migration script executed without errors but the search index needs reindexing.",
     "The migration script executed without errors, but the search index needs reindexing.",
     "Use a comma before coordinating conjunctions separating independent clauses.", "engineering"),
    ("The client approved the revised wireframes and our designers began prototyping.",
     "The client approved the revised wireframes, and our designers began prototyping.",
     "Use a comma before 'and' when joining two independent clauses.", "client_handling"),
    ("I wrapped up the documentation updates so we are ready for the security review.",
     "I wrapped up the documentation updates, so we are ready for the security review.",
     "Use a comma before 'so' connecting independent clauses.", "workplace_chitchat"),
    ("We can deploy the hotfix today or we can wait for tomorrow's maintenance window.",
     "We can deploy the hotfix today, or we can wait for tomorrow's maintenance window.",
     "Use a comma before 'or' between independent clauses.", "engineering"),

    # Question marks on inquiries
    ("Could you please confirm if Friday at 2 PM works for the project demo.",
     "Could you please confirm if Friday at 2 PM works for the project demo?",
     "Direct inquiries require a question mark at the end.", "client_handling"),
    ("Do we have sufficient test coverage for the payment webhook handler.",
     "Do we have sufficient test coverage for the payment webhook handler?",
     "Use a question mark for interrogative sentences.", "engineering"),
    ("Would you be interested in joining our team lunch this Wednesday.",
     "Would you be interested in joining our team lunch this Wednesday?",
     "Use a question mark at the end of an invitation inquiry.", "workplace_chitchat"),
    ("Should we scale up the Kubernetes replicas ahead of the marketing campaign launch.",
     "Should we scale up the Kubernetes replicas ahead of the marketing campaign launch?",
     "Interrogative sentences must end with a question mark.", "engineering"),
    ("Can you send over the updated statement of work by this afternoon.",
     "Can you send over the updated statement of work by this afternoon?",
     "Requests framed as questions require a question mark.", "client_handling"),

    # Apostrophes in contractions
    ("We dont expect any downtime during the blue-green deployment phase.",
     "We don't expect any downtime during the blue-green deployment phase.",
     "Add an apostrophe in 'don't'.", "engineering"),
    ("Please let us know if you cant access the shared Google Drive folder.",
     "Please let us know if you can't access the shared Google Drive folder.",
     "Add an apostrophe in 'can't'.", "client_handling"),
    ("Im heading into a meeting now, but ill review your pull request right after.",
     "I'm heading into a meeting now, but I'll review your pull request right after.",
     "Add apostrophes in 'I'm' and 'I'll'.", "workplace_chitchat"),
    ("Thats exactly what we needed to solve the deadlock issue.",
     "That's exactly what we needed to solve the deadlock issue.",
     "Add an apostrophe in 'that's'.", "workplace_chitchat"),
    ("Lets schedule a quick standup to unblock the frontend integration.",
     "Let's schedule a quick standup to unblock the frontend integration.",
     "Add an apostrophe in 'let's'.", "workplace_chitchat"),
    ("Theres no need to worry about backward compatibility; version two supports all legacy endpoints.",
     "There's no need to worry about backward compatibility; version two supports all legacy endpoints.",
     "Add an apostrophe in contraction 'there's'.", "engineering"),
    ("Weve verified the database backups across all regions.",
     "We've verified the database backups across all regions.",
     "Add an apostrophe in contraction 'we've'.", "engineering"),

    # Possessive apostrophes
    ("We reviewed the clients feedback and prioritized their top three feature requests.",
     "We reviewed the client's feedback and prioritized their top three feature requests.",
     "Use an apostrophe for singular possessive 'client's'.", "client_handling"),
    ("All users data is encrypted at rest using AES-256 standard encryption.",
     "All users' data is encrypted at rest using AES-256 standard encryption.",
     "Use an apostrophe after the 's' for plural possessive 'users''.", "engineering"),
    ("The team velocity has increased significantly over the last three sprints.",
     "The team's velocity has increased significantly over the last three sprints.",
     "Use an apostrophe for possessive 'team's'.", "engineering"),
    ("The companys policy requires two-factor authentication for all remote access.",
     "The company's policy requires two-factor authentication for all remote access.",
     "Use singular possessive 'company's'.", "client_handling"),

    # Semicolons and Colons
    ("The unit tests passed however the integration tests reported timeouts.",
     "The unit tests passed; however, the integration tests reported timeouts.",
     "Use a semicolon before 'however' and a comma after it when linking clauses.", "engineering"),
    ("The client approved the design therefore we can proceed to sprint planning.",
     "The client approved the design; therefore, we can proceed to sprint planning.",
     "Use a semicolon before 'therefore' and a comma after it.", "client_handling"),
    ("The release contains three major enhancements faster indexing clean UI and better caching.",
     "The release contains three major enhancements: faster indexing, clean UI, and better caching.",
     "Use a colon to introduce an explanatory list.", "engineering"),

    # Compound modifier hyphens
    ("TypoFormer operates as a real time keystroke companion on standard CPUs.",
     "TypoFormer operates as a real-time keystroke companion on standard CPUs.",
     "Hyphenate compound adjectives preceding a noun: 'real-time'.", "engineering"),
    ("Our engineers deployed a low latency message queue across multiple regions.",
     "Our engineers deployed a low-latency message queue across multiple regions.",
     "Hyphenate compound modifier 'low-latency'.", "engineering"),
    ("We are preparing client facing documentation for the upcoming platform release.",
     "We are preparing client-facing documentation for the upcoming platform release.",
     "Hyphenate compound modifier 'client-facing'.", "client_handling"),
    ("The architecture achieves zero downtime rolling updates in Kubernetes.",
     "The architecture achieves zero-downtime rolling updates in Kubernetes.",
     "Hyphenate compound modifier 'zero-downtime'.", "engineering"),
    ("We implemented an in memory cache using Redis to minimize disk reads.",
     "We implemented an in-memory cache using Redis to minimize disk reads.",
     "Hyphenate compound adjective 'in-memory'.", "engineering"),
    ("This is a mission critical system that requires automated failover.",
     "This is a mission-critical system that requires automated failover.",
     "Hyphenate compound modifier 'mission-critical'.", "engineering"),
    ("We maintain an open source repository for community contributions.",
     "We maintain an open-source repository for community contributions.",
     "Hyphenate compound modifier 'open-source'.", "engineering"),

    # Proper nouns & Acronyms
    ("We wrote the data ingestion script in python using the polars library.",
     "We wrote the data ingestion script in Python using the Polars library.",
     "Capitalize proper programming language and library names.", "engineering"),
    ("The microservices are containerized using docker and orchestrated with kubernetes.",
     "The microservices are containerized using Docker and orchestrated with Kubernetes.",
     "Capitalize proprietary platform names: Docker, Kubernetes.", "engineering"),
    ("Please submit your changes via a pull request on github.",
     "Please submit your changes via a pull request on GitHub.",
     "Capitalize proper brand name 'GitHub'.", "engineering"),
    ("The backend runs on an aws EC2 instance configured with linux.",
     "The backend runs on an AWS EC2 instance configured with Linux.",
     "Capitalize acronyms and OS names: AWS, Linux.", "engineering"),
    ("All communications with the api server must utilize https and return json.",
     "All communications with the API server must utilize HTTPS and return JSON.",
     "Capitalize technical acronyms: API, HTTPS, JSON.", "engineering"),
    ("Our devops engineer configured automated ci/cd pipelines.",
     "Our DevOps engineer configured automated CI/CD pipelines.",
     "Capitalize 'DevOps' and acronym 'CI/CD'.", "engineering"),
]

# =============================================================================
# 6. SENTENCE STRUCTURE & SYNTACTIC AGREEMENT SCENARIOS
# =============================================================================
GRAMMAR_SYNTAX_SCENARIOS = [
    # Subject-verb agreement
    ("The collection of microservices are monitored continuously by Datadog.",
     "The collection of microservices is monitored continuously by Datadog.",
     "The singular subject 'collection' takes the singular verb 'is'.", "engineering"),
    ("The cluster of Kubernetes worker nodes have recovered from the network partition.",
     "The cluster of Kubernetes worker nodes has recovered from the network partition.",
     "The singular subject 'cluster' requires the singular verb 'has'.", "engineering"),
    ("Neither the frontend engineer nor the backend developer were able to replicate the bug.",
     "Neither the frontend engineer nor the backend developer was able to replicate the bug.",
     "With 'neither... nor' connecting singular subjects, use singular 'was'.", "engineering"),
    ("The latency of our edge inference models have dropped below ten milliseconds.",
     "The latency of our edge inference models has dropped below ten milliseconds.",
     "The head noun 'latency' is singular and takes 'has'.", "engineering"),
    ("A comprehensive list of deliverables and milestones were sent to the stakeholder.",
     "A comprehensive list of deliverables and milestones was sent to the stakeholder.",
     "The singular subject 'list' takes 'was'.", "client_handling"),
    ("Each of the microservice endpoints require authentication tokens.",
     "Each of the microservice endpoints requires authentication tokens.",
     "'Each' is singular and takes singular verb 'requires'.", "engineering"),
    ("The set of unit tests for authentication fail intermittently under load.",
     "The set of unit tests for authentication fails intermittently under load.",
     "Subject 'set' is singular and takes 'fails'.", "engineering"),
    ("Every one of our developers have reviewed the security policy.",
     "Every one of our developers has reviewed the security policy.",
     "'Every one' is singular and takes 'has'.", "workplace_chitchat"),

    # Tense consistency
    ("Yesterday the team conducts a post-mortem and identified the root cause.",
     "Yesterday the team conducted a post-mortem and identified the root cause.",
     "Use past tense 'conducted' to match 'yesterday' and 'identified'.", "engineering"),
    ("We reviewed the client feedback last week and resolve all open questions.",
     "We reviewed the client feedback last week and resolved all open questions.",
     "Maintain past tense 'resolved' across coordinated past actions.", "client_handling"),
    ("Earlier this morning, Fahim pushes the hotfix and verified the staging deployment.",
     "Earlier this morning, Fahim pushed the hotfix and verified the staging deployment.",
     "Use past tense 'pushed' for completed past actions.", "workplace_chitchat"),
    ("Last sprint we deploy the new billing portal and migrated fifty accounts.",
     "Last sprint we deployed the new billing portal and migrated fifty accounts.",
     "Use past tense 'deployed' for completed sprint actions.", "engineering"),

    # Prepositions in technical / business English
    ("Our cloud architecture complies to the highest enterprise security standards.",
     "Our cloud architecture complies with the highest enterprise security standards.",
     "The standard preposition with 'comply' is 'with'.", "engineering"),
    ("Our senior DevOps engineer is responsible of managing the Kubernetes cluster.",
     "Our senior DevOps engineer is responsible for managing the Kubernetes cluster.",
     "Use 'responsible for', not 'responsible of'.", "engineering"),
    ("The client expressed that they are very interested for exploring the enterprise tier.",
     "The client expressed that they are very interested in exploring the enterprise tier.",
     "Use 'interested in', not 'interested for'.", "client_handling"),
    ("We will focus at reducing response latency during the upcoming sprint.",
     "We will focus on reducing response latency during the upcoming sprint.",
     "The standard collocation is 'focus on', not 'focus at'.", "engineering"),
    ("The new release is completely different than the legacy version.",
     "The new release is completely different from the legacy version.",
     "Standard English favors 'different from'.", "engineering"),
    ("Our platform consists with twelve independently scalable microservices.",
     "Our platform consists of twelve independently scalable microservices.",
     "Use 'consists of', not 'consists with'.", "engineering"),
    ("Prior from the deployment, we verified that all integration tests passed.",
     "Prior to the deployment, we verified that all integration tests passed.",
     "Use 'prior to', not 'prior from'.", "engineering"),

    # Concise word choices / Professional phrasing
    ("We need to refactor this module in order to improve maintainability.",
     "We need to refactor this module to improve maintainability.",
     "Simplify wordy 'in order to' to 'to'.", "engineering"),
    ("At this point in time, all API endpoints are operating within normal parameters.",
     "Currently, all API endpoints are operating within normal parameters.",
     "Replace wordy 'At this point in time' with 'Currently'.", "client_handling"),
    ("Due to the fact that the server ran out of memory, the process was killed.",
     "Because the server ran out of memory, the process was killed.",
     "Replace 'Due to the fact that' with 'Because'.", "engineering"),
    ("In the event that the primary database fails, automatic failover triggers within seconds.",
     "If the primary database fails, automatic failover triggers within seconds.",
     "Replace wordy 'In the event that' with 'If'.", "engineering"),
]

# =============================================================================
# 7. PRISTINE CLEAN ANCHOR SENTENCES (38% OF DATASET)
# Zero-hallucination GEC anchor & Autocomplete Next-Word Corpus
# =============================================================================
CLEAN_ENGINEERING_ANCHORS = [
    "We decoupled the event processing pipeline to achieve sub-millisecond end-to-end latency.",
    "The pull request includes comprehensive unit tests and passes all continuous integration checks.",
    "Our distributed database utilizes raft consensus to ensure consistency across multiple availability zones.",
    "The quantized INT8 model runs efficiently on edge CPUs with minimal memory overhead.",
    "Please profile the memory allocations to verify that no circular references exist in the cache.",
    "We implemented exponential backoff with jitter on all outgoing HTTP client requests.",
    "The Kubernetes ingress controller routes traffic based on path prefixes and hostname headers.",
    "All sensitive configuration secrets are securely retrieved from HashiCorp Vault at runtime.",
    "We upgraded the PostgreSQL connection pool to prevent thread exhaustion under peak load.",
    "The character-aware CNN embedding layer enables robust inference even on unknown tokens.",
    "Our team established strict code review guidelines to maintain architectural consistency across services.",
    "The microservice communicates asynchronously via Apache Kafka topics partitioned by tenant ID.",
    "We automated the deployment pipeline using GitHub Actions with branch protection rules.",
    "The Redis cache stores precomputed embedding vectors to accelerate nearest-neighbor search.",
    "We configured Prometheus metrics and Grafana alerts to monitor CPU and memory utilization.",
    "The system guarantees at-least-once delivery semantics for all critical transactional events.",
    "We performed a benchmark comparison between the FP32 baseline and the quantized ONNX graph.",
    "The API gateway validates JWT bearer tokens and enforces rate limits per client IP address.",
    "We refactored the legacy monolith into modular packages with clean public interfaces.",
    "The load balancer automatically drains active connections before taking an instance offline.",
    "We verified that all database queries utilize appropriate composite indexes for fast execution.",
    "The new asynchronous worker pool reduced our background job processing latency by half.",
    "Our continuous integration pipeline automatically builds and verifies multi-platform Docker containers.",
    "The caching layer invalidates stale records whenever a mutation webhook is received.",
    "We added extensive integration tests to cover edge cases in our token quantization routine.",
    "The GraphQL schema defines strict types and input validations for all client mutations.",
    "We implemented mutual TLS between internal services to enforce zero-trust security principles.",
    "The system scales horizontally by adding worker replicas as queue depth exceeds thresholds.",
    "We optimized the memory footprint so the binary comfortably runs on client desktop machines.",
    "All service logs are structured in JSON format and indexed into Elasticsearch for rapid debugging.",
]

CLEAN_CLIENT_ANCHORS = [
    "Thank you for sharing your feedback; we have incorporated the revisions into the latest release.",
    "We are pleased to confirm that all milestone deliverables have been completed on schedule.",
    "Our engineering team is actively investigating the issue, and we will provide an update within the hour.",
    "Please find attached the detailed project roadmap and resource allocation breakdown for your review.",
    "We appreciate your partnership and look forward to presenting the live demonstration on Thursday.",
    "Could you please confirm if the proposed implementation timeline aligns with your business goals?",
    "We have scheduled a dedicated training session next week to assist your team with the onboarding process.",
    "Our enterprise agreement includes twenty-four-seven technical support and guaranteed response times.",
    "We conducted thorough acceptance testing to ensure seamless compatibility with your existing systems.",
    "I would be delighted to schedule a follow-up discussion to address any outstanding questions you may have.",
    "Thank you for your patience while we finalized the customized reporting modules for your dashboard.",
    "We are committed to delivering a robust, high-performance solution that scales alongside your growth.",
    "Please let us know if you require any adjustments to the scope of work prior to contract signing.",
    "The new feature set will be rolled out to your staging environment by close of business tomorrow.",
    "We value your input and have prioritized the requested security enhancements in the upcoming sprint.",
    "We are pleased to present the quarterly performance summary and platform uptime statistics.",
    "Our technical account manager will be your primary point of contact throughout the migration phase.",
    "Please review the attached statement of work and let us know if you have any questions.",
    "We have completed the requested custom integrations and validated them against your sandbox environment.",
    "Thank you for your continued trust in our team; we look forward to achieving our shared milestones.",
    "We are excited to kick off Phase Two of the project following the successful delivery of Phase One.",
    "Could you kindly confirm the attendees from your side for the executive briefing on Wednesday?",
    "Our support team has successfully resolved the reported ticket and verified normal operations.",
]

CLEAN_CHITCHAT_ANCHORS = [
    "Hey Fahim, do you have a few minutes for a quick huddle regarding the schema migration?",
    "Awesome job on the architecture presentation today; the stakeholders were very impressed!",
    "I will be stepping away for lunch and will be back online in about thirty minutes.",
    "Good morning everyone! Let's make sure to update our sprint board before the daily standup.",
    "Thanks so much for helping me debug that tricky race condition yesterday afternoon.",
    "Let me know whenever you're free to review the updated pull request on GitHub.",
    "I just pushed the hotfix to staging, so feel free to test it whenever it's convenient for you.",
    "Are you planning to attend the engineering all-hands meeting later this afternoon?",
    "Have a fantastic weekend everyone, and thank you for all your hard work on this release!",
    "Quick heads-up: I'm currently reviewing the PR and will leave my comments within the hour.",
    "Happy Friday team! We managed to clear all critical backlog tickets for this sprint.",
    "Let's sync up after the customer demo to debrief on their feedback and next steps.",
    "I noticed you optimized the build pipeline—it's running noticeably faster now!",
    "Could you share the link to the meeting notes when you get a chance?",
    "No worries at all; take your time and ping me whenever you're ready.",
    "Sorry about that; I was speaking while muted! Can everyone hear me clearly now?",
    "I'll be working remotely tomorrow morning but easily reachable on Slack if anything comes up.",
    "Congratulations to the whole team on shipping the new release ahead of schedule!",
    "Does anyone want to grab coffee before we head into the sprint planning session?",
    "That makes total sense to me; let's proceed with the proposed implementation approach.",
    "I'm heading out for the day; feel free to ping me asynchronously if anything urgent arises.",
    "Thanks for the thorough code review feedback; I've addressed all of your comments in the latest commit.",
    "Hope you had a relaxing and refreshing weekend! Ready to tackle the new sprint goals.",
]


def generate_wine_dataset(
    target_count: int = 28000,
    clean_ratio: float = 0.38,
    seed: int = 42,
) -> list[WineRow]:
    random.seed(seed)
    rows: list[WineRow] = []
    row_id = 1

    # 1. Clean Anchor Sentences (38% of total dataset)
    clean_target = int(target_count * clean_ratio)
    print(f"Generating {clean_target:,} pristine clean anchor sentences...")

    clean_pool = (
        CLEAN_ENGINEERING_ANCHORS * 200 +
        CLEAN_CLIENT_ANCHORS * 200 +
        CLEAN_CHITCHAT_ANCHORS * 200
    )
    random.shuffle(clean_pool)

    for text in clean_pool[:clean_target]:
        if any(w in text for w in ["pull request", "latency", "Kubernetes", "cache", "microservice", "SQL", "Docker", "API", "pipeline", "PostgreSQL", "GraphQL"]):
            domain = "engineering"
        elif any(w in text for w in ["client", "milestone", "roadmap", "partnership", "agreement", "deliverables", "stakeholder", "SLA"]):
            domain = "client_handling"
        else:
            domain = "workplace_chitchat"

        rows.append(WineRow(
            id=f"WINE_{row_id:06d}",
            input_text=text,
            target_text=text,
            error_type="clean_anchor",
            error_category="Clean",
            error_count=0,
            correction_explanation="Sentence is grammatically correct, well-punctuated, and fluent.",
            domain=domain,
            complexity_level="advanced" if domain == "engineering" else "intermediate",
        ))
        row_id += 1

    # 2. Homophone & Confused Word Scenarios (~22%)
    print("Generating homophone and vocabulary confusion scenarios...")
    homophone_sources = TECH_HOMOPHONES * 85 + CLIENT_HOMOPHONES * 75 + CHITCHAT_HOMOPHONES * 70
    random.shuffle(homophone_sources)

    for wrong, right, tpl, explanation, domain in homophone_sources:
        input_sentence = tpl.format(word=wrong)
        target_sentence = tpl.format(word=right)
        rows.append(WineRow(
            id=f"WINE_{row_id:06d}",
            input_text=input_sentence,
            target_text=target_sentence,
            error_type="vocabulary_confusion",
            error_category="Vocabulary",
            error_count=1,
            correction_explanation=explanation,
            domain=domain,
            complexity_level="intermediate",
        ))
        row_id += 1
        if len(rows) >= int(target_count * 0.60):
            break

    # 3. Punctuation & Capitalization Scenarios (~20%)
    print("Generating punctuation, capitalization, and apostrophe scenarios...")
    punct_sources = PUNCTUATION_SCENARIOS * 115
    random.shuffle(punct_sources)

    for inp, tgt, explanation, domain in punct_sources:
        rows.append(WineRow(
            id=f"WINE_{row_id:06d}",
            input_text=inp,
            target_text=tgt,
            error_type="punctuation",
            error_category="Punctuation",
            error_count=1,
            correction_explanation=explanation,
            domain=domain,
            complexity_level="fundamental" if "Capitalize" in explanation else "intermediate",
        ))
        row_id += 1
        if len(rows) >= int(target_count * 0.80):
            break

    # 4. Syntactic Agreement & Grammar Scenarios (~10%)
    print("Generating syntactic agreement, tense, and preposition scenarios...")
    grammar_sources = GRAMMAR_SYNTAX_SCENARIOS * 125
    random.shuffle(grammar_sources)

    for inp, tgt, explanation, domain in grammar_sources:
        rows.append(WineRow(
            id=f"WINE_{row_id:06d}",
            input_text=inp,
            target_text=tgt,
            error_type="grammar_agreement",
            error_category="Grammar",
            error_count=1,
            correction_explanation=explanation,
            domain=domain,
            complexity_level="advanced",
        ))
        row_id += 1
        if len(rows) >= int(target_count * 0.90):
            break

    # 5. Natural Spelling Slips (NO CROSS-PRODUCT GIBBERISH)
    print("Generating natural spelling slips in domain-appropriate sentences...")
    typo_sources = NATURAL_SPELLING_PAIRS * 55
    random.shuffle(typo_sources)

    for wrong, right, tpl, domain in typo_sources:
        if len(rows) >= target_count:
            break
        input_sentence = tpl.format(word=wrong)
        target_sentence = tpl.format(word=right)
        rows.append(WineRow(
            id=f"WINE_{row_id:06d}",
            input_text=input_sentence,
            target_text=target_sentence,
            error_type="spelling_slip",
            error_category="Spelling",
            error_count=1,
            correction_explanation=f"Correct typo '{wrong}' to '{right}'.",
            domain=domain,
            complexity_level="fundamental",
        ))
        row_id += 1

    random.shuffle(rows)
    print(f"Total 'Wine' dataset synthesized: {len(rows):,} rows.")
    return rows


def save_and_split_dataset(
    rows: list[WineRow],
    output_dir: Path,
    val_ratio: float = 0.10,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    master_csv = output_dir / "wine_dataset.csv"
    train_csv = output_dir / "wine_train.csv"
    val_csv = output_dir / "wine_val.csv"
    lm_clean_jsonl = output_dir / "wine_lm_clean.jsonl"

    headers = [
        "id", "input_text", "target_text", "error_type", "error_category",
        "error_count", "correction_explanation", "domain", "complexity_level"
    ]

    print(f"Saving Master CSV -> {master_csv}...")
    with master_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(list(r))

    n_val = int(len(rows) * val_ratio)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    print(f"Saving Train CSV ({len(train_rows):,} rows) -> {train_csv}...")
    with train_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in train_rows:
            writer.writerow(list(r))

    print(f"Saving Val CSV ({len(val_rows):,} rows) -> {val_csv}...")
    with val_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in val_rows:
            writer.writerow(list(r))

    clean_texts = [r.target_text for r in rows if r.target_text]
    seen = set()
    dedup_clean: list[str] = []
    for t in clean_texts:
        if t not in seen:
            seen.add(t)
            dedup_clean.append(t)

    print(f"Saving {len(dedup_clean):,} clean sentences for Causal LM Autocomplete -> {lm_clean_jsonl}...")
    with lm_clean_jsonl.open("w", encoding="utf-8") as f:
        for t in dedup_clean:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    return master_csv, train_csv, val_csv, lm_clean_jsonl


def main():
    dataset_dir = Path("data/dataset")
    rows = generate_wine_dataset(target_count=28000, clean_ratio=0.38)
    save_and_split_dataset(rows, dataset_dir)


if __name__ == "__main__":
    main()
