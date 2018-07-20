import subprocess
import os


class LanguageToolWrapper:
    """
    LanguageTool API NLP UK wrapper (https://github.com/brown-uk/nlp_uk)

    """

    @staticmethod
    def run_tag_text_utility(tagtext_dir_path, input_file, output_file):
        """
        groovy TagText.groovy -i <input_file> -o <output_file>
        -l - одна лексема на рядок (лише для виводу в txt)
        -f - залишає тільки першу лему        
        """

        cmd = ['groovy', tagtext_dir_path + 'TagText.groovy', '-i', input_file, '-o', output_file, '-f']
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outs, errs = process.communicate()

        print('Outputs: ', outs)
        print('Errors: ', errs)

    @staticmethod
    def run_clean_text_utility(cleantext_tool_dir_path, input_directory):
        """
        Читає всі файли .txt в теці й намагається знайти ті, що відповідають крітеріям українського тексту
        (напр. задана мінімальна кількість українських слів, наразі 80), вивід йде в <txt_dir>/good/

        Перед застосуванням критерію намагається почистити і виправити типові проблеми:

        поломані/мішані кодування
        мішанину латиниці та кирилиці
        нетипові апострофи (міняє на прямий — ')
        вилучає м'який дефіс (00AD)
        об'єднує перенесення слів на новий рядок (використовуючи орфографічний словник)

        CleanText.groovy[ < txt_dir >] {-wc < min_word_limit >}(типово: txt /)

        """

        if os.path.exists(input_directory + '/good' + '/corpus.txt'):
            os.remove(input_directory + '/good' + '/corpus.txt')

        cmd = ['groovy', cleantext_tool_dir_path + 'CleanText.groovy', input_directory, '{-wc 80}']
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outs, errs = process.communicate()

        print('Outputs: ', outs)
        print('Errors: ', errs)


