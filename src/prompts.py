def create_prompt_qa(dataset, prompt_type, question,  evidence=[], celebrities=''):
    if prompt_type in ['base', 'celebrity']:
        prompt = f'You are a helpful assistant for journalism. Your task is to predict the {question} of the given news image.\n' 
        if celebrities!='':
            prompt += f'The following public figures can be seen in the image {celebrities}.\n'
            print(prompt)
        if len(evidence)!=0:
            prompt += f"Leverage the image's content and relevant additional information to predict the {question}.\n" 
            prompt += "The following news articles might be relevant to the events shown in the image. Use them to answer the question in addition to the image's content. They are sorted by order of relevance:\n\n"
            for p in evidence:
                for  item in [('Publication date: ', 'date'), ('Location tags: ', 'location'), ('Abstract: ', 'abstract'), ('Content: ', 'lead_paragraph')]:
                    prompt += f"{item[0]}{p[item[1]]}\n" 
                prompt+='\n'
        else:
            prompt += f"Leverage the image's content to predict the {question}.\n"


        if question=='location':
            prompt += 'Where was the image taken?\nAnswer only with the city, region, and country, structured as a comma-separted list (city,region,country).'
        elif question=='time':
            #The time limits differ for TARA and 5Pils-OOC
            prompt += 'When was the image taken?\nAnswer only with a date (yyyy-mm-dd, yyyy--mm, or yyyy), as specific as possible.'
            if dataset=='tara':
                prompt+= f'The date need to be included in the range [1900-01-01, 2021-12-31].'
            else:
                #5Pils OOC
                prompt += f'The date need to be included in the range [1900-01-01, 2023-12-31].'
    else:
        prompt="You are an expert detective. You specialize in analyzing a scene and reasoning beyond the scene. For a given image, you will look at the scene and all the probable inferences and reason based on them to reason beyond the image to retrieve your best guess for the event that is going on or has happened in this image, the background related to this image, the geospatial information and the temporal information. \n\ntemporal information:\n- century\n- decade\n- year\n- month\n- day\n geospatial information:\n   - country\n   - state_or_province\n   - city\nevent : the event in this image\nbackground: the most relative background related to the event\n\nYou will return the output in the required response format. It is absolutely imperative that you return the JSON output. You will extract the information required for the response format. If no plausible guess can be made for a field, output NA."
    return prompt


def create_prompt_captioning(input_text, num_output):
    demo=  [ 
                {
                    "demo_input": "In the midst of a real estate crash, Dubai pulled out all the stops to celebrate the opening of the world’s tallest building. Burdened by debt and a devastating real estate crash, Dubai is doing what it does best: doubling down.",
                    "demo_output_0": "A breathtaking view of the Burj Khalifa illuminated with dazzling lights during its grand opening ceremony in Dubai.",
                    "demo_output_1": "A visitor gets a view of Dubai from the 124th floor of Burj Khalifa, the world’s tallest building, on Monday.",
                    "demo_output_2": "A close-up of the Burj Khalifa's shimmering exterior, showcasing the architectural marvel of the world’s tallest building.",
                    "demo_output_3": "Crowds gather in awe at the base of the Burj Khalifa as fireworks light up the night sky during its inauguration.",
                    "demo_output_4": "An aerial shot of Dubai’s downtown area, highlighting the stark contrast between the glittering Burj Khalifa and the surrounding real estate developments."
                },
                {
                    "demo_input": "The announcement by the Connecticut Democrat opens the way for the state’s popular attorney general to run for his Senate seat. EAST HADDAM, Conn. Christopher J. Dodd has been in politics 36 years, including an ill-starred presidential run in 2008 and a record three decades as a United States senator from Connecticut. He can, then, count votes with anyone in Congress. But with his popularity at a nadir, he was not sure he could count enough votes in his home state to assure his election to a sixth term.",
                    "demo_output_0": "Senator Christopher J. Dodd announces his decision not to seek re-election during a press conference in East Haddam, Connecticut.",
                    "demo_output_1": "Senator Christopher J. Dodd, Democrat of Connecticut,   on Wednesday with his wife, Jackie, and their daughter Christina.",
                    "demo_output_2": "The state’s attorney general, a potential candidate for Dodd's Senate seat, speaks at a community event following Dodd's announcement.",
                    "demo_output_3": "Christopher J. Dodd, a five-term Connecticut senator, speaks to reporters about his decision to step down after 36 years in politics.",
                    "demo_output_4": "A reflective moment for Senator Christopher J. Dodd as he acknowledges his political legacy during his retirement announcement."
                },
                {
                    "demo_input": "A suicide bomber aiming for a pro-government militia commander detonated his bomb-laden vest in the provincial capital of Gardez on Thursday. KABUL, Afghanistan A suicide bomber attacking a pro-government militia commander detonated his bomb-laden vest in a southeastern provincial capital, Gardez, on Thursday, and witnesses said he killed 10 people and wounded 27, most of them civilians. Also on Thursday, the governor of a neighboring province survived a bomb attack.",
                    "demo_output_0": "Emergency responders and civilians gather at the site of the suicide bombing in Gardez, Afghanistan, which left 10 dead and 27 injured.",
                    "demo_output_1": "Afghan security forces inspect the aftermath of the attack in Gardez, with debris scattered across the blast site.",
                    "demo_output_2": "Families grieve outside a hospital in Gardez as casualties from the suicide bombing continue to arrive.",
                    "demo_output_3": "The provincial capital of Gardez witnesses heightened security as officials investigate the suicide bombing targeting a pro-government militia commander.",
                    "demo_output_4": "Damaged storefronts and shattered windows reflect the devastation caused by the bombing in Gardez on Thursday."
                },
                {
                    "demo_input": "The charges against six military officers involved in the ouster of Manuel Zelaya are expected to be dropped as part of a deal to ease tensions. MEXICO CITY Six military officers involved in the ouster of Manuel Zelaya from the Honduran presidency last year were charged this week with abuse of power, but the charges are expected to be dropped as part of a deal to ease tensions in the country, officials said.",
                    "demo_output_0": "Honduran military officers accused of abuse of power leave the courtroom as tensions surrounding the ouster of Manuel Zelaya persist.",
                    "demo_output_1": "Supporters of Manuel Zelaya protest outside a government building, demanding accountability for the military-led ouster.",
                    "demo_output_2": "A courtroom in Honduras where six military officers face charges related to the removal of President Manuel Zelaya.",
                    "demo_output_3": "A Honduran flag flies outside the Supreme Court in Tegucigalpa as the country grapples with the aftermath of last year’s political crisis.",
                    "demo_output_4": "Security forces stand guard near a protest in Tegucigalpa as officials discuss dropping charges against military officers involved in Zelaya's ouster."
                }
            
            ]
  
    prompt = f"You are a helpful assistant for journalism. You are given the headline of a news article and your task is to generate {num_output} news image captions that would be suitable to illustrate this news article. Answer only with the captions as a list of strings.\n\n"


    for d in demo:
        demo_output = []
        prompt+= f"News article headline: {d['demo_input']}\n\n"
        prompt+= "News image captions:\n"
        for i in range(num_output):
            demo_output.append(d[f"demo_output_{i}"])
        prompt+= str(demo_output) + "\n\n"
    prompt+= f"News article headline: {input_text}\n\n"
    prompt+= "News image captions:\n"
    
    return prompt


def create_prompt_article_classification(input_text):
    demo=   [ 
                {
                    "demo_input": "In the midst of a real estate crash, Dubai pulled out all the stops to celebrate the opening of the world’s tallest building. Burdened by debt and a devastating real estate crash, Dubai is doing what it does best: doubling down.",
                    "demo_output": "Category 1",
                },
                {
                    "demo_input": "About $2.3 billion in new biopharmaceutical manufacturing plants were going up in the Boston area. Companies battling for an edge in the biopharmaceutical industry have $2.3 billion in manufacturing plants in development in the Boston area to produce genetically engineered drugs.",
                    "demo_output": "Category 2",
                },
                {
                    "demo_input": " A human rights group distributed video cameras to young Gazans and asked them to tell about their lives. GAZA In the year since Israeli fighter jets and troops invaded this coastal Palestinian strip to stop rocket fire, time seems to have stood still. A blockade imposed by both Israel and Egypt to isolate the Hamas government bars the vast majority of goods and people from moving in or out. That means there is no reconstruction of destroyed buildings. Thousands remain homeless. Winter has arrived.",
                    "demo_output": "Category 1",
                },
                {
                    "demo_input": "On nearly every front, President Obama's goal of lower deficits has gotten harder since his first budget a year ago. WASHINGTON President Obama is making final decisions on his budget for next year and is still promising to outline a path to substantially lower federal deficits. But on nearly every front, that goal has gotten harder since his first budget a year ago.",
                    "demo_output": "Category 2",
                },
                {
                    "demo_input": "A 32-year-old Iranian citizen had procured cyanide and ricin in an effort “to commit an Islamist-motivated attack,” the authorities said. BERLIN — Late Saturday night, specialized police descended on a calm commercial strip in the town of Castrop-Rauxel, in western Germany.",
                    "demo_output": "Category 1",

                },
                {
                    "demo_input": "The finullcial dealings of Governor-elect Christopher J. Christie’s brother could present pitfalls in Trenton. If he weren't the younger brother of New Jersey’s newly elected governor, Todd J. Christie might be seen as just another freewheeling, risk-taking, deep-pocketed Wall Street trader.",
                    "demo_output": "Category 2",
                },
                {
                    "demo_input": "An anti-whaling group said Wednesday that the new high-speed boat it was using to harass Japanese whalers has been badly damaged in a violent collision at sea.",
                    "demo_output": "Category 1",
                },
                {
                    "demo_input": "The Orthodox Christmas service at the Pechersky Lavra monastery in Kyiv was held for the first time by the Ukrainian-led branch of the church, rather than the Russian-led one.",
                    "demo_output": "Category 1"
                }
            ]

    prompt = f"You are a helpful assistant for journalism. You are given the headline of a news article which comes with an image. Your task is to classify the news article in one of the two categories based on the content of the headline.\nCategory 1:  the content discusses at least one visual event, i.e., a physical, observable event, and it is very likely that the image  accompanying the news article is showing this event.\nCategory 2: the news article is not discussing any visual event, e.g., it discusses only political decisions or the results of the stock exchange, it is an interview, ... it is likely that the image accompanying the article serves the role of a stock image. In other words, the image is likely not at the core of the news article.\n\n"
    #add demos
    for d in demo:
        prompt+= f"News article headline: {d['demo_input']}\n\n"
        prompt+= "Answer only with 'Category 1' or 'Category 2':\n"
        prompt+= str(d['demo_output']) + "\n\n"
    #Start of prompt for input
    prompt+= f"News article headline: {input_text}\n\n"
    prompt+= "Answer only with 'Category 1' or 'Category 2':\n"
    return prompt