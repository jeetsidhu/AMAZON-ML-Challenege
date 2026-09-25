threshold 0.79; true pairs 385020; links 376189

| false negatives | n | share of true pairs | indic | alias | domain | no address | S1 has namesakes | S3 share |
|---|---|---|---|---|---|---|---|---|
| retrieval_miss | 4541 | 0.0118 | 0.089 | 0.005 | 0.056 | 0.717 | 0.812 | 0.562 |
| outranked | 1879 | 0.0049 | 0.006 | 0.005 | 0.011 | 0.947 | 0.907 | 0.514 |
| below_threshold | 2968 | 0.0077 | 0.031 | 0.005 | 0.018 | 0.569 | 0.667 | 0.521 |
| found | 375632 | 0.9756 | 0.074 | 0.021 | 0.052 | 0.027 | 0.391 | 0.516 |

base rates over all true pairs: {"all_true_pairs": {"f_indic": 0.07381175003895901, "f_alias": 0.02105085450106488, "f_domain": 0.051602514155108824, "f_noaddr": 0.0439873253337489}, "s1_has_namesakes": 0.4001921978079061}

| false positives | n | share of links | indic | alias | domain | no address | S1 has namesakes | mean p |
|---|---|---|---|---|---|---|---|---|
| decoy_linked | 412 | 0.0011 | 0.029 | 0.002 | 0.029 | 0.068 | 0.308 | 0.935 |
| wrong_entity | 145 | 0.0004 | 0.145 | 0.000 | 0.028 | 0.510 | 0.552 | 0.896 |

### retrieval_miss
   {"s1": "My Big Hospitality Private Limited", "s1_addr": "Plot No 4, Row House, Adoni Ypr Home Bypass, Kurnool, Andhra Pradesh", "record": "#mybig", "record_addr": "KURNOOL, Andhra Pradesh, 4", "p_true": null, "p_best": 0.09890270978212357}
   {"s1": "Shree Global Private Limited", "s1_addr": "Pl -35, S No 24 25 Shikshak Nagar Wanwadi, Pune, Maharashtra, Pune", "record": "श्री ग्लोबल प्राइवेट लिमिटेड", "record_addr": "PL -3-5, PUNE, Maharashtra", "p_true": null, "p_best": 2.0728404706460424e-05}
   {"s1": "Swastik Marketing Private Limited", "s1_addr": "No 2, Shree Laksmi Nilaya, 4Th Cross, 24Th Main Road, Jp Nagar, 5Th Phase, Near Big Market, Bangalore, Karnataka", "record": "ಸ್ವಸ್ತಿಕ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "record_addr": "H.NO 2, BANGALORE, Karnataka", "p_true": null, "p_best": 9.716308704810217e-05}
   {"s1": "Laxmi Foods Private Limited", "s1_addr": "29/033A, Near Oel House, Rana Pratap Marg Hazratganj, Lucknow, Uttar Pradesh", "record": "लक्ष्मी फूड्स प्राइवेट लिमिटेड", "record_addr": "Uttar Pradesh, 29/033A, CHANDRAWAL, NEAR OEL HOUSE, LUCKNOW", "p_true": null, "p_best": 0.0054534743539988995}
   {"s1": "Mickle & Brooks Verde LLC", "s1_addr": "11 Mulberry Lane, Manchester, CT", "record": "Mickle & Brooks Verde", "record_addr": null, "p_true": null, "p_best": 0.3823908567428589}
   {"s1": "Durga College Pvt Ltd", "s1_addr": "131, 2Nd Floor Hargovind Enclave, New Delhi, East Delhi, Delhi", "record": "Durga Co1lege Pvt Ltd", "record_addr": null, "p_true": null, "p_best": 0.27720725536346436}

### outranked
   {"s1": "ONE Summit LP", "s1_addr": "721 Owasso Avenue, Tulsa, OK", "record": "ONE SUMMIT LP", "record_addr": null, "p_true": 0.2955109477043152, "p_best": 0.5429037809371948}
   {"s1": "Jsk Plastic Partners", "s1_addr": "108, Pandav Nagar, Agra, Uttar Pradesh", "record": "Jsk Plastic Partners Corp", "record_addr": null, "p_true": 0.43023374676704407, "p_best": 0.44361191987991333}
   {"s1": "American Fund Corp", "s1_addr": "89 Washington Street, St. Joe, IN", "record": "American Fund Corporation", "record_addr": null, "p_true": 0.23969309031963348, "p_best": 0.3091462552547455}
   {"s1": "Calcutta & Brothers Limited", "s1_addr": "H.No 4, First Floor, Mount View Villa, Chunabhatti, Bhopal, Madhya Pradesh", "record": "Mr Calcutta Brothers Límited Center", "record_addr": null, "p_true": 0.25121191143989563, "p_best": 0.2752031087875366}
   {"s1": "Ivenex Nevada", "s1_addr": "406 Jackson Avenue, Lisbon, ND", "record": "Ivenex Nevada", "record_addr": null, "p_true": 0.3736788332462311, "p_best": 0.5741959810256958}
   {"s1": "House Consulting Private Limited", "s1_addr": "Third Floor, Survey Nos.12/1, 12/2A & 13/1A, Challaghatta Village, Var, Thur Hobli, Karnataka, Bangalore, Divyasree Greens, Bangalore", "record": "HOUSE CONSULTING PRIVATE LIMITED", "record_addr": null, "p_true": 0.0687054693698883, "p_best": 0.4147331714630127}

### below_threshold
   {"s1": "Vision Technologies Pvt Ltd", "s1_addr": "Plot No - 350/436, Khata No - 180/105, Bhubaneswar, Khordha, Orissa", "record": "ଭିଜନ୍ ଟେକ୍ନୋଲୋଜିସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍", "record_addr": "PLOT NO - 350/436, KHORDHA, Odisha", "p_true": 0.7814674377441406, "p_best": 0.7814674377441406}
   {"s1": "Great Systems Private Limited", "s1_addr": "Seat U, C-1, S/F, L- Type, Block- C, Subhash Park, Uttam Nagar, New Delhi, West Delhi, Delhi", "record": "Sri Great Systems Private", "record_addr": "DOOR NO 019 SEAT U, C-1, S/F, L- TYPE, BLOCK- C, SUBHASH PARK, UTTAM NAGAR, NEW DELHI, Delhi", "p_true": 0.6601516604423523, "p_best": 0.6601516604423523}
   {"s1": "Technologies Solitude Dispatch Limited", "s1_addr": "C/O, Arun Prakash Harnandka, 127, N.S Road, 1St Floor, Room No. 1/2, Kolkata, Kolkata, Howrah, West Bengal", "record": "LIMITED TECHNOLOGIES SOLITUDE DÍSPATCH", "record_addr": "C/O, NORTH 24 PARAGANAS, HOWRAH, West Bengal", "p_true": 0.13219836354255676, "p_best": 0.13219836354255676}
   {"s1": "Boucher Aeon", "s1_addr": "3544 Jacob Street, Wheeling, WV", "record": "TAVOVIO #51958", "record_addr": "Jacob St, WHEELING, WV", "p_true": 0.13776953518390656, "p_best": 0.13776953518390656}
   {"s1": "Fitzgerald Regional Yield", "s1_addr": "21347 Starling Drive, Bend, OR", "record": "Fitzgerald Regional", "record_addr": null, "p_true": 0.733788251876831, "p_best": 0.733788251876831}
   {"s1": "Penn Capital", "s1_addr": "4049 75th Avenue, Saint Cloud, MN", "record": "Penn Capital Inc.", "record_addr": "4050 75TH AVE, MN, SAINT CLOUD, PMB 6143", "p_true": 0.7322333455085754, "p_best": 0.7322333455085754}

### decoy_linked
   {"s1": "Hashmi, Clotilda W., MD, DDS PC", "s1_addr": "7813 Shady Banks Terrace, Chesterfield County, VA", "record": "Hashmi, Clotilda W., MMD, DDS", "record_addr": "7813 SHADY BANKS TER, CHESTERFIELD COUNTY, VA", "p": 0.9999476075172424}
   {"s1": "HD Asset-Mumbai", "s1_addr": "Anand Niwas 4Th Rdrajawadi Ghatkopar East, Mumbai, Maharashtra", "record": "GD ASSET-MUMBAI", "record_addr": "ANAND NIWAS 4-4TH RDRAJAWADI GHATKOPAR EAST, MUMBAI, Maharashtra", "p": 0.999905526638031}
   {"s1": "Sela Gutierrez, MD", "s1_addr": "8754 Scarlet Oak, Willis, TX", "record": "The Sela Gutierrez, HMD", "record_addr": "WILLIS, TX, 8754 SCARLET OAK", "p": 0.9998928904533386}
   {"s1": "Yamuavi Prism LLP", "s1_addr": "House No.40/328, Unnat Nagar, M.G.Road, Goregaon West, Goregaon West, Mumbai, Maharashtra", "record": "LLP Yamuaex Prsnm", "record_addr": "House No.40/328, Unnat Nagar, M.g.road, Goregaon West, Goregaon West, Mumbai, महाराष्ट्र", "p": 0.999868631362915}
   {"s1": "Patriot Center VI", "s1_addr": "1767 Siebert Way, Fl 0, Grants Pass, OR", "record": "PATRIOT CENTER XVI", "record_addr": "GRANTS PASS CITY, 1767 SIEBERT WAY, OR", "p": 0.9998668432235718}
   {"s1": "Madhuram India Pvt. Ltd.", "s1_addr": "17/2, Dollar Chamber, Lal Bagh Road, Bangalore, Karnataka", "record": "Madhuram Open Pvt. (Limited)", "record_addr": "17/2, Dollar Chamber, Lal Bagh Road, Bangalore, Bengaluru, KA", "p": 0.9998542666435242}

### wrong_entity
   {"s1": "Dream Global Private Limited", "s1_addr": "C/O Mr. Dattatraya Maruti Marne, Flat No. 2 1St Floor, Swapna Sarkar, Shinde Nagar, Kothrud, Pune, Maharashtra", "record": "ड्रीम ग्लोबल प्राइवेट लिमिटेड", "record_addr": "FLAT NO. 02, BOMBAY, Maharashtra", "p": 0.9996609687805176}
   {"s1": "Campus Tissue Ltd", "s1_addr": "C 803, Dream Apartments, Plot No.14 Sector 22, Dwarka, Delhi, New Delhi, Delhi", "record": "SSN", "record_addr": "C 803, Dream Apartments, Plot No.14 Sector 22, Dwarka, Delhi, New Delhi, Delhi", "p": 0.999033510684967}
   {"s1": "Shiv Foods Private Limited", "s1_addr": "Maharashtra, Nagpur, Somalwadawardha Road, Nagpur", "record": "Private Shiv S0lutions [Limited]", "record_addr": "C/o, Nagpur, MH", "p": 0.9985801577568054}
   {"s1": "Bansari (India) Solutions Private Limited", "s1_addr": "Srr Exurbia, Villa # 36, Anekal, Bangalore, Karnataka", "record": "solutionssuper.com", "record_addr": "Villa 36, Bangalore, Bangalore South, KA", "p": 0.998526394367218}
   {"s1": "Jain Producer Private Limited", "s1_addr": "Second Floor, Greater Kailash I, New Delhi, Delhi, # S-222, New Delhi", "record": "जैन प्रोड्यूसर प्राइवेट लिमिटेड", "record_addr": "#10 SECOND FLOOR, NEW DELHI, WEST DELHI, Delhi", "p": 0.9982492923736572}
   {"s1": "Blue Aditya Services Pvt Ltd", "s1_addr": "Kucha Ustad Dag, 3Rd Floor, Chandni Chowk, 4933, North Delhi, Delhi", "record": "Blue Aditya Private Limited Services", "record_addr": null, "p": 0.9976800680160522}
