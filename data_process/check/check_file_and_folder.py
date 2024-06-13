import os
import re
import glob
import argparse
import tqdm
import soundfile as sf
import difflib

def add_space(match):
    char = match.group()
    return char + ' ' if char[-1] not in '()' else char

def check_wujiqiao_dir(root_path):
    # Traverse all first-level directories
    print(root_path)
    for level1_dir_name in next(os.walk(root_path))[1]:
        level1_dir_path = os.path.join(root_path, level1_dir_name)
        # Traverse all second-level directories
        for level2_dir_name in next(os.walk(level1_dir_path))[1]:
            level2_dir_path = os.path.join(level1_dir_path, level2_dir_name)
            # Check if the list of third-level directories under the second-level directory contains "Control"
            if "Control" not in next(os.walk(level2_dir_path))[1]:
                print("Error: " + level2_dir_path + " does not have wjq")
                return False
    return True

def check_file_exist(wujiqiao_dir, tech_dir, tech):
    files1 = os.listdir(wujiqiao_dir)
    files2 = os.listdir(tech_dir)

    files1_replaced = [file.replace('Control', tech) for file in files1]
    files2_replaced = [file.replace(tech, 'Control') for file in files2]

    for file in files1_replaced:
        if file not in files2 and '.wav' in file:
            print(f"File does not exist error:\ncompared to wjq dirs \n\033[92m{wujiqiao_dir}\033[0m\n\033[93m{file}\033[0m \nnot in \033[92m{tech_dir}\033[0m\n")
    for file in files2_replaced:
        if file not in files1 and '.wav' in file:
            print(f"wjq file does not exist error:\ncompared to tech dirs \n\033[92m{tech_dir}\033[0m\n\033[93m{file}\033[0m \nnot in \033[92m{wujiqiao_dir}\033[0m\n")

def check_read_file_exist(read_dir, wujiqiao_dir, tech):
    files1 = os.listdir(read_dir)
    files2 = os.listdir(wujiqiao_dir)

    files1_replaced = [file.replace('Speech', 'Control') for file in files1]
    files2_replaced = [file.replace('Control', 'Speech') for file in files2]

    for file in files1_replaced:
        if file not in files2 and '.wav' in file:
            print(f"wjq file does not exist error:\ncompared to reading dirs \n\033[92m{read_dir}\033[0m\n\033[93m{file}\033[0m \nnot in \033[92m{wujiqiao_dir}\033[0m\n")
    for file in files2_replaced:
        if file not in files1 and '.wav' in file:
            print(f"Reading file does not exist error:\ncompared to wjq dirs \n\033[92m{wujiqiao_dir}\033[0m\n\033[93m{file}\033[0m \nnot in \033[92m{read_dir}\033[0m\n")

def check_file_structure(root_path, sex):
    for song_name in sorted((glob.glob(f"{root_path}/*/*"))):
        tech = song_name.split('/')[-2]
        if tech == "True and False Mixed" or tech == "True and False":
            wujiqiao_dir = os.path.join(song_name, "Control")
            tech_dir1 = os.path.join(song_name, "Falsetto")
            tech_dir2 = os.path.join(song_name, "Mixed")
            check_file_exist(wujiqiao_dir, tech_dir1, 'Falsetto')
            check_file_exist(wujiqiao_dir, tech_dir2, 'Mixed')
        else:
            wujiqiao_dir = os.path.join(song_name, "Control")
            tech_dir = os.path.join(song_name, tech)
            check_file_exist(wujiqiao_dir, tech_dir, tech)
        wujiqiao_dir = os.path.join(song_name, "Control")
        read_dir = os.path.join(song_name, "Speech")
        check_read_file_exist(read_dir, wujiqiao_dir, 'Control')

def check_wav_name(root_path, foreign_flag=False):
    for wav_name in sorted((glob.glob(f"{root_path}/*/*/*/*.wav"))):
        wav_name_base = os.path.basename(wav_name)
        wav_name_split = wav_name_base[:-4].split("_")
        if wav_name_split[0] not in ['Male', 'Female']:
            print("Gender Error: " + wav_name)
            return False
        tech_name = wav_name.split("/")[-2]
        if wav_name_split[1] != tech_name:
            print("Tech_name Error: " + wav_name)
            return False
        if foreign_flag:
            txt_name = wav_name.replace(".wav", ".txt")
            if tech_name != "Speech" and tech_name != "Recitation":
                if not os.path.exists(txt_name):
                    print(f'Error: the wav\'s txt does not exist, expected: {txt_name}')
                    return False
            if os.path.exists(txt_name):
                with open(txt_name, 'r') as f:
                    txt_content = f.read()
                    if txt_content == "":
                        print(f'Error: {txt_name} is empty')
                        return False
    return True

def rename_files_and_folders(root_path):
    for path, dirs, files in os.walk(root_path, topdown=False):
        for name in files:
            if args.french and ' .wav' in name:
                new_name = name.replace(' .wav','.wav')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                files[files.index(name)] = new_name
                print(f"Rename File: {name} -> {new_name}")
        for name in files:
            if args.french and ' .txt' in name:
                new_name = name.replace(' .txt','.txt')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                files[files.index(name)] = new_name
                print(f"Rename File: {name} -> {new_name}")
        for name in files:
            if '２' in name:
                new_name = name.replace('２','2')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                files[files.index(name)] = new_name
                print(f"Rename File: {name} -> {new_name}")
        for name in dirs:
            if ' ' in name:
                new_name = name.replace(' ','_')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                dirs[dirs.index(name)] = new_name
                print(f"Rename Directory: {name} -> {new_name}")
        for name in files:
            if ' ' in name:
                new_name = name.replace(' ','_')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                files[files.index(name)] = new_name
                print(f"Rename File: {name} -> {new_name}")
        for name in dirs:
            if '‘' in name:
                new_name = name.replace('‘', '')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                dirs[dirs.index(name)] = new_name
                print(f"Rename Directory: {name} -> {new_name}")

        # change real to Control
        for name in files:
            if 'real' in name:
                new_name = name.replace('-', '_').replace('real','Control')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                files[files.index(name)] = new_name
                print(f"Rename File: {name} -> {new_name}")
        # change Mixed to Mixed_Voice
        for name in files:
            if 'Mix' in name:
                new_name = name.replace('-', '_').replace('Mix','Mixed_Voice')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                files[files.index(name)] = new_name
                print(f"Rename File: {name} -> {new_name}")
        for name in dirs:
            if 'Mix' in name:
                new_name = name.replace('-', '_').replace('Mix','Mixed_Voice')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                dirs[dirs.index(name)] = new_name
                print(f"Rename Directory: {name} -> {new_name}")
        # change Vibrato to Vibrato
        for name in files:
            if 'Vibra' in name:
                new_name = name.replace('-', '_').replace('Vibra','Vibrato')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                files[files.index(name)] = new_name
                print(f"Rename File: {name} -> {new_name}")
        for name in dirs:
            if 'Vibra' in name:
                new_name = name.replace('-', '_').replace('Vibra','Vibrato')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                dirs[dirs.index(name)] = new_name
                print(f"Rename Directory: {name} -> {new_name}")
        # change Breathe to Breathy
        for name in files:
            if 'Breathe' in name:
                new_name = name.replace('Breathe','Breathy')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                files[files.index(name)] = new_name
                print(f"Rename File: {name} -> {new_name}")
        for name in dirs:
            if 'Breathe' in name:
                new_name = name.replace('Breathe','Breathy')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                dirs[dirs.index(name)] = new_name
                print(f"Rename Directory: {name} -> {new_name}")
       
        # 将-替换为_
        for name in files:
            if '-' in name:
                new_name = name.replace('-','_')
                os.rename(os.path.join(path, name), os.path.join(path,new_name))
                files[files.index(name)] = new_name   
                print(f"Rename File: {name} -> {new_name}")
        for name in files:
            if ' ' in name:
                if(not args.french):
                    new_name = name.replace(' ','_')
                    os.rename(os.path.join(path, name), os.path.join(path,new_name)) 
                    files[files.index(name)] = new_name  
                    print(f"Rename File: {name} -> {new_name}")
                else:
                    new_name = name.replace(' ','')
                    os.rename(os.path.join(path, name), os.path.join(path,new_name)) 
                    files[files.index(name)] = new_name  
                    print(f"Rename File: {name} -> {new_name}")

def rename_xml(root_path, sex):
    for xml_name in sorted((glob.glob(f"{root_path}/*/*/*/*.musicxml"))):
        tech_name = xml_name.split('/')[-2]
        base_name = os.path.basename(xml_name)
        if sex == True:
            new_name = xml_name.replace(base_name, f"女声_{tech_name}.musicxml")
            if xml_name == new_name:
                continue
            os.rename(xml_name, new_name)
            print(f"rename the musicxml file：{base_name} -> {os.path.basename(new_name)}")
        else:
            new_name = xml_name.replace(base_name, f"男声_{tech_name}.musicxml")
            if xml_name == new_name:
                continue
            os.rename(xml_name, new_name)
            print(f"rename the musicxml file：{base_name} -> {os.path.basename(new_name)}")
    return

def rename_txt(root_path, sex):
    for txt_name in sorted((glob.glob(f"{root_path}/*/*/*/*.txt"))):
        tech_name = txt_name.split('/')[-2]
        base_name = os.path.basename(txt_name)
        number = base_name.split('_')[-1][:-4]
        # print(number)
        if("句"in number):
            if sex == True:
                new_name = txt_name.replace(base_name, f"女声_{tech_name}_{number}.txt")
                if txt_name == new_name:
                    continue
                os.rename(txt_name, new_name)
                print(f"rename the txt file：{base_name} -> {os.path.basename(new_name)}")
            else:
                new_name = txt_name.replace(base_name, f"男声_{tech_name}_{number}.txt")
                if txt_name == new_name:
                    continue
                os.rename(txt_name, new_name)
                print(f"rename the txt file：{base_name} -> {os.path.basename(new_name)}")
        else:
            if sex == True:
                new_name = txt_name.replace(base_name, f"女声_{tech_name}.txt")
                if txt_name == new_name:
                    continue
                os.rename(txt_name, new_name)
                print(f"rename the txt file：{base_name} -> {os.path.basename(new_name)}")
            else:
                new_name = txt_name.replace(base_name, f"男声_{tech_name}.txt")
                if txt_name == new_name:
                    continue
                os.rename(txt_name, new_name)
                print(f"rename the txt file：{base_name} -> {os.path.basename(new_name)}")
    return      

def modify_content(file):
    with open(file, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        content = re.sub(r'[\uac00-\ud7a3]', add_space, content) # for korean
        content = re.sub(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', add_space, content) # for japanese
        # content = re.sub(r'(\d) ', r'\1', content)
        content = content.replace('…', '')
        content = content.replace('。', '')
        content = content.replace('?', ' ')
        content = content.replace('¿', ' ')
        content = content.replace('，', ',')
        content = content.replace('（', '(')
        content = content.replace('）', ')')
        content = content.replace('\n',' ')
        content = content.replace('\x0c','')
        content = content.replace('’','\'').replace('‘',"'")
        content = content.replace('\xa0',' ')
        content = content.replace('\u3000',' ').replace('\u2002',' ').replace('\u2003',' ').replace('\u2005',' ').replace('\u200b',' ').replace('\u205f', ' ').replace('\u2028', ' ')
        content = content.replace(' )',')')
        content = content.replace('\t','')
        if(args.german):
            content = content.replace('\n', '')
            if('Control'not in file):
                content = content.replace(' ', '')
        if(args.japan):
            content = content.replace('(ce)', '')
        flag = 0
        for i in range(len(content)):
            if content[i] == '(':
                flag += 1
            if content[i] == ')':
                flag -= 1
            if flag not in [0, 1]:
                print(f"error about () is in {file}")
                print(content)
                print(i,content[i-10:i+10])
                break
            if content[i] == ',' and flag == 0:
                content = content[:i] + ' ' + content[i+1:]
        # check if there are continuous spaces
        i = 0
        while i < len(content) - 1:
            if content[i] == ' ' and content[i+1] == ' ':
                content = content[:i] + content[i+1:]
            else:
                i += 1

        # if there is a space before `(`, remove it
        i = 1
        while i < len(content):
            if content[i] == '(' and content[i-1] == ' ':
                content = content[:i-1] + content[i:]
                i -= 1  # ignore the space just removed
            i += 1

        # if there is no space after `)`, add one
        i = 0
        while i < len(content) - 1:
            if content[i] == ')' and content[i+1] != ' ':
                content = content[:i+1] + ' ' + content[i+1:]
                i += 1  # ignore the space just added
            i += 1
        
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

def batch_modify_content(data_dir):
    files = sorted((glob.glob(f"{data_dir}/*/*/*/*.txt")))
    for file in tqdm.tqdm(files):
        modify_content(file)
    print("all the txt files has been adjusted!")

def get_word_list(file):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        word_list = content.split(' ')
        word_list = list(filter(lambda x: x, word_list))
        ret_word_list = [element.split('(')[0] for element in word_list]

    return ret_word_list

def check_words(root_path):
    for txt_name in tqdm.tqdm(sorted((glob.glob(f"{root_path}/*/*/*/*.txt")))):
        if "Speech" not in txt_name and "朗诵" not in txt_name and "Control" not in txt_name:
            tech_word_list = get_word_list(txt_name)
            # get the corresponding control txt file
            tech_base_name = os.path.basename(txt_name)
            # print(tech_base_name)
            sex_name = tech_base_name.split('_')[0]
            number_name = tech_base_name.split('_')[-1]
            wujiqiao_base_name = f'{sex_name}_Control_{number_name}'if len(tech_base_name.split('_'))==3 else f'{sex_name}_Control.txt'
            # tech_word_list = get_word_list(txt_name)
            txt_path_list = txt_name.split('/')
            wujiqiao_path = '/'.join(txt_path_list[:-2]) + '/Control/' + wujiqiao_base_name

            wujiqiao_word_list = get_word_list(wujiqiao_path)
            if tech_word_list != wujiqiao_word_list:
                print(f"Error:\t {txt_name} \nand\t {wujiqiao_path}\nhave different words\n")
                if(len(tech_word_list) != len(wujiqiao_word_list)):
                    print(f'len(tech_word_list): \033[93m{len(tech_word_list)}\033[0m, len(wujiqiao_word_list): \033[93m{len(wujiqiao_word_list)}\033[0m\n')
                    list1 = tech_word_list
                    list2 = wujiqiao_word_list
                    s = difflib.SequenceMatcher(None, tech_word_list, wujiqiao_word_list)
                    output1 = []
                    output2 = []
                    for tag, i1, i2, j1, j2 in s.get_opcodes():
                        if tag == 'replace':
                            output1.extend(['\033[93m' + item + '\033[0m' for item in list1[i1:i2]])
                            output2.extend(['\033[93m' + item + '\033[0m' for item in list2[j1:j2]])
                        elif tag == 'delete':
                            output1.extend(['\033[93m' + item + '\033[0m' for item in list1[i1:i2]])
                            output2.extend([''] * (i2-i1))
                        elif tag == 'insert':
                            output1.extend([''] * (j2-j1))
                            output2.extend(['\033[93m' + item + '\033[0m' for item in list2[j1:j2]])
                        elif tag == 'equal':
                            output1.extend(['\033[92m' + item + '\033[0m' for item in list1[i1:i2]])
                            output2.extend(['\033[92m' + item + '\033[0m' for item in list2[j1:j2]])

                    print("tech:")
                    for item in output1:
                        print(item, end=' ')
                    print("\n")
                    print("wjq:")
                    for item in output2:
                        print(item, end=' ')
                    print("\n")
                    print(tech_word_list)
                    print(wujiqiao_word_list)
                else:
                    for i in range(len(tech_word_list)):
                        if tech_word_list[i] != wujiqiao_word_list[i]:
                            print(f"{i}, \ntech:\t \033[93m{tech_word_list[i]}\033[0m, \nwjq:\t \033[93m{wujiqiao_word_list[i]}\033[0m")
                

def check_wav_time(root_path):
    # check the wav time between technique and control
    print('start checking wav time...')
    for wav_name in tqdm.tqdm(sorted((glob.glob(f"{root_path}/*/*/*/*.wav")))):
        if "Speech" not in wav_name and "Control" not in wav_name:
            tech_base_name = os.path.basename(wav_name)
            sex_name = tech_base_name.split('_')[0]
            number_name = tech_base_name.split('_')[-1]
            wujiqiao_base_name = f'{sex_name}_Control_{number_name}' if len(tech_base_name.split('_'))==3 else f'{sex_name}_Control.wav'
            wav_path_list = wav_name.split('/')
            wujiqiao_path = '/'.join(wav_path_list[:-2]) + '/Control/' + wujiqiao_base_name
            # print(wav_name,wujiqiao_path)

            tech_time = sf.info(wav_name).duration
            wujiqiao_time = sf.info(wujiqiao_path).duration
            if abs(tech_time - wujiqiao_time) > 5:
                print(f"Error: \n{'/'.join(wujiqiao_path.split('/')[wujiqiao_path.split('/').index('Singers'):wujiqiao_path.split('/').index('Control')])} 目录下 \n {os.path.basename(wav_name)} \033[93m %.2fs\033[0m\n {os.path.basename(wujiqiao_path)} \033[93m%.2fs\033[0m \n时间差异过大，需要重录\n"%(tech_time,wujiqiao_time))
    print('All wavs pass the test!')
    return True
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', type=str, required=True, help="Root directory path")
    parser.add_argument('--female', action='store_true', help="Is it female voice?")
    parser.add_argument('--foreign', action='store_true', help='Is it a foreign language?')
    parser.add_argument('--german', action='store_true', help='Is it German?')
    parser.add_argument('--japan', action='store_true', help='Is it Japanese?')
    parser.add_argument('--french', action='store_true', help='Is it French?')
    parser.add_argument('--korean', action='store_true', help='Is it Korean?')
    args = parser.parse_args()
    # Explanation: Chinese female voice, Chinese male voice, foreign female voice, foreign male voice
    if args.female and args.foreign:
        print("############# Foreign Female Voice #######################")
    if not args.female and args.foreign:
        print("############# Foreign Male Voice #######################")
    if args.female and not args.foreign:
        print("############# Chinese Female Voice #######################")
    if not args.female and not args.foreign:
        print("############# Chinese Male Voice #######################") 
    
    print(check_wujiqiao_dir(args.root_path))
    print(check_wav_name(args.root_path, args.foreign))
    rename_files_and_folders(args.root_path)
    print(check_wav_name(args.root_path, args.foreign))
    check_wav_time(args.root_path)
    if args.foreign:
        rename_txt(args.root_path, args.female)
        print("All txt files have been renamed successfully!\n")
        check_file_structure(args.root_path, args.female)
        print("Start to check and modify contents in txt files...")
        batch_modify_content(args.root_path)
        print("Start to check the words in txt files...")
        check_words(args.root_path)
        print("Finished the words' checking!")
    else:
        rename_xml(args.root_path, args.female)
        print("All xml files have been renamed successfully!\n")
        