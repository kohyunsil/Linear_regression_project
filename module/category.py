def production (row):
    disney = ['Walt Disney Animation Studios', 'Walt Disney Pictures', 'Pixar Animation Studios']
    searchlight = ['Fox Searchlight Pictures']
    fox = ['20th Century Fox Film Corporation', 'Twentieth Century Fox', 'Twentieth Century Fox Animation']
    marvel = ['Marvel Studios', 'Marvel Entertainment', 'Marvel Enterprises']
    dreamworks = ['DreamWorks', 'DreamWorks Animation']
    atlas = ['Atlas Entertainment', 'Atlas Productions', 'Atlas Independent']
    pathe = ['Path챕', 'Path챕 Pictures International']
    liongate = ['Lionsgate', 'Lionsgate Premiere']
    warnerbros = ['Warner Bros.', 'Warner Bros. Pictures', 'Warner Bros. Digital Distribution', 'Warner Animation Group',
                  'Warner Bros. Animation']
    twodux2 = ['2DUX짼']
    sony = ['Sony Pictures Classics', 'Sony Pictures Entertainment (SPE)', 'Sony Pictures Animation']
    paramount = ['Paramount Players', 'Paramount Pictures', 'Paramount Vantage', 'Paramount Animation']
    
    if row in disney:
        return 'Walt Disney'
    if row in searchlight:
        return 'Searchlight Pictures'
    if row in fox:
        return 'Twentieth Century Fox'
    if row in marvel:
        return 'Marvel Studios'
    if row in dreamworks:
        return 'DreamWorks'
    if row in atlas:
        return 'Atlas Entertainment'
    if row in pathe:
        return 'Pathe'
    if row in liongate:
        return 'Lionsgate'
    if row in warnerbros:
        return 'Warner Bros.'
    if row in twodux2:
        return '2DUX2'
    if row in sony:
        return 'Sony Pictures'
    if row in paramount:
        return 'Paramount Pictures'
    else:
        return row


def month(row):
    if row == 1:
        return 'Jan'
    if row == 2:
        return 'Feb'
    if row == 3:
        return 'Mar'
    if row == 4:
        return 'Apr'
    if row == 5:
        return 'May'
    if row == 6:
        return 'Jun'
    if row == 7:
        return 'Jul'
    if row == 8:
        return 'Aug'
    if row == 9:
        return 'Sep'
    if row == 10:
        return 'Oct'
    if row == 11:
        return 'Nov'
    if row == 12:
        return 'Dec'

def season(row):
    winter = ['Jan', 'Feb', 'Dec']
    spring = ['Mar', 'Apr', 'May']
    summer = ['Jun', 'Jul', 'Aug']
    fall = ['Sep', 'Oct', 'Nov']
    if row in winter:
        return 'Winter'
    if row in spring:
        return 'Spring'
    if row in summer:
        return 'Summer'
    if row in fall:
        return 'Fall'


def baseonbooks (row):
    if row == 0:
        return "Not based on books"
    if row == 1:
        return "base on books"

def academy (row) :
    Actor = ['ACTOR', 'ACTOR IN A LEADING ROLE']
    Actor_support = ['ACTOR IN A SUPPORTING ROLE']
    Actress = ['ACTRESS', 'ACTRESS IN A LEADING ROLE']
    Actress_support = ['ACTRESS IN A SUPPORTING ROLE']
    Cinematography = ['CINEMATOGRAPHY', 'CINEMATOGRAPHY (Black-and-White)',
                     'CINEMATOGRAPHY (Color)']
    Directing = ['DIRECTING (Comedy Picture)', 'DIRECTING (Dramatic Picture)', 'DIRECTING']
    Best_Picture = ['OUTSTANDING PICTURE', 'OUTSTANDING PRODUCTION', 'OUTSTANDING MOTION PICTURE',
                    'BEST MOTION PICTURE', 'BEST PICTURE']
    Writing = ['WRITING (Adaptation)', 'WRITING (Original Story)', 'WRITING (Title Writing)',
               'WRITING', 'WRITING (Screenplay)', 'WRITING (Original Screenplay)',
               'WRITING (Original Motion Picture Story)', 'WRITING (Motion Picture Story)',
               'WRITING (Story and Screenplay)', 'WRITING (Screenplay--Adapted)', 'WRITING (Screenplay--Original)',
               'WRITING (Screenplay--based on material from another medium)',
               'WRITING (Story and Screenplay--written directly for the screen)',
               'WRITING (Story and Screenplay--based on material not previously published or produced)',
               'WRITING (Story and Screenplay--based on factual material or material not previously published or produced)',
               'WRITING (Screenplay Adapted from Other Material)','WRITING (Screenplay Written Directly for the Screen--based on factual material or on story material not previously published or produced)',
               'WRITING (Screenplay Based on Material from Another Medium)',
               'WRITING (Screenplay Written Directly for the Screen)',
               'WRITING (Screenplay Based on Material Previously Produced or Published)',
               'WRITING (Adapted Screenplay)']
    etc = ['ART DIRECTION', 'ENGINEERING EFFECTS', 'UNIQUE AND ARTISTIC PICTURE', 'SPECIAL AWARD',
           'SOUND RECORDING', 'SHORT SUBJECT (Cartoon)', 'SHORT SUBJECT (Comedy)',
           'SHORT SUBJECT (Novelty)', 'ASSISTANT DIRECTOR', 'FILM EDITIN', 'MUSIC (Scoring)',
           'MUSIC (Song)', 'DANCE DIRECTION', 'SHORT SUBJECT (Color)', 'SHORT SUBJECT (One-reel)',
           'SHORT SUBJECT (Two-reel)', 'IRVING G. THALBERG MEMORIAL AWARD', 'MUSIC (Original Score)',
           'SPECIAL EFFECTS', 'ART DIRECTION (Black-and-White)', 'ART DIRECTION (Color)',
           'DOCUMENTARY (Short Subject)', 'MUSIC (Music Score of a Dramatic Picture)',
           'MUSIC (Scoring of a Musical Picture)', 'DOCUMENTARY', 'MUSIC (Music Score of a Dramatic or Comedy Picture)',
           'DOCUMENTARY (Feature)', 'COSTUME DESIGN (Black-and-White)', 'COSTUME DESIGN (Color)', 
           'SPECIAL FOREIGN LANGUAGE FILM AWARD', 'HONORARY FOREIGN LANGUAGE FILM AWARD',
           'HONORARY AWARD','FOREIGN LANGUAGE FILM', 'JEAN HERSHOLT HUMANITARIAN AWARD', 'COSTUME DESIGN', 'SHORT SUBJECT (Live Action)',
           'SOUND', 'MUSIC (Music Score--substantially original)', 'MUSIC (Scoring of Music--adaptation or treatment)',
           'SOUND EFFECTS', 'SPECIAL VISUAL EFFECTS', 'MUSIC (Original Music Score)',
           'MUSIC (Original Score--for a motion picture [not a musical])',
           'MUSIC (Score of a Musical Picture--original or adaptation)', 
           'MUSIC (Song--Original for the Picture)','MUSIC (Original Song Score)', 
           'MUSIC (Original Dramatic Score)', 'MUSIC (Scoring: Adaptation and Original Song Score)',
           'SHORT SUBJECT (Animated)', 'SPECIAL ACHIEVEMENT AWARD (Visual Effects)',
           'MUSIC (Scoring: Original Song Score and Adaptation -or- Scoring: Adaptation)',
           'SHORT FILM (Animated)', 'SHORT FILM (Live Action)', 'MUSIC (Original Song)',
           'SPECIAL ACHIEVEMENT AWARD (Sound Effects)', 'MUSIC (Original Song Score and Its Adaptation or Adaptation Score)',
           'VISUAL EFFECTS', 'SPECIAL ACHIEVEMENT AWARD', 'SPECIAL ACHIEVEMENT AWARD (Sound Effects Editing)',
           'MUSIC (Adaptation Score)', 'MUSIC (Original Song Score and Its Adaptation -or- Adaptation Score)',
           'SPECIAL ACHIEVEMENT AWARD (Sound Editing)', 'SHORT FILM (Dramatic Live Action)',
           'MAKEUP', 'SOUND EFFECTS EDITING', 'MUSIC (Original Song Score or Adaptation Score)',
           'MUSIC (Original Musical or Comedy Score)', 'SOUND EDITING',
           'ANIMATED FEATURE FILM', 'SOUND MIXING', 'MAKEUP AND HAIRSTYLING', 
           'PRODUCTION DESIGN', 'INTERNATIONAL FEATURE FILM']
         
        
    if row in Actor:
        return "ACTOR"
    if row in Actor_support:
        return "ACTOR SUPPORTING"
    if row in Actress:
        return "ACTRESS"
    if row in Actress_support:
        return "ACTRESS SUPPORTING"
    if row in Cinematography:
        return "CINEMATOGRAPHY"
    if row in Directing:
        return "DIRECTING"
    if row in Best_Picture:
        return "BEST PICTURE"
    if row in Writing:
        return "WRITING"
    if row in etc:
        return "ETC"