import os
import sys  # NOQA
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir))
from cmd2 import Cmd, options
from optparse import make_option
from python.models import model_based_reccomendation, svd_on_transaction_data, svd_on_movie_lens, als_spark
class InteractiveCommands(Cmd):

    @options([make_option('-n', '--noOfItems', type=int, help="No of recommendation to be generated"),
              make_option('-r', '--refresh', type=str, help="Generate fresh recommendation"),
              make_option('-u', '--userid', type=int, help="Enter the userid")])
    def do_knn_on_movielens(self,args, opts):
        if not (opts.noOfItems):
            raise Exception('mandatory parameters no of items missing')
        if not (opts.userid):
            raise Exception('mandatory parameters userid missing')
        if not(opts.refresh) or opts.refresh != "yes":
            opts.refresh="no"
        if int(opts.noOfItems):
            model_based_reccomendation.do_knn_on_movie_lens(opts.userid,opts.noOfItems,opts.refresh)
        else:
            print("Supply a valid value of no of items")

    @options([make_option('-n', '--noOfItems', type=int, help="No of recommendation to be generated"),
              make_option('-r', '--refresh', type=str, help="Generate fresh recommendation"),
              make_option('-u', '--userid', type=int, help="Enter the userid")])
    def do_svd_on_movielens(self,args, opts):
        if not (opts.noOfItems):
            raise Exception('mandatory parameters no of items missing')
        if not (opts.userid):
            raise Exception('mandatory parameters userid missing')
        if not(opts.refresh) or opts.refresh != "yes":
            opts.refresh="no"
        if int(opts.noOfItems):
            svd_on_movie_lens.do_svd_on_movie_lens(opts.userid,opts.noOfItems,opts.refresh)
        else:
            print("Supply a valid value of no of items")

    @options([make_option('-n', '--noOfItems', type=int, help="No of recommendation to be generated"),
              make_option('-r', '--refresh', type=str, help="Generate fresh recommendation"),
              make_option('-u', '--userid', type=int, help="Enter the userid")])
    def do_als_based_recommendation(self,args, opts):
        if not (opts.noOfItems):
            raise Exception('mandatory parameters no of items missing')
        if not (opts.userid):
            raise Exception('mandatory parameters userid missing')
        if not(opts.refresh) or opts.refresh != "yes":
            opts.refresh="no"
        if int(opts.noOfItems):
            als_spark.start_spark_recommendation(opts.userid, opts.noOfItems, opts.refresh)
        else:
            print("Supply a valid value of no of items")

    @options([make_option('-n', '--noOfItems', type=int, help="No of recommendation to be generated"),
              make_option('-r', '--refresh', type=str, help="Generate fresh recommendation"),
              make_option('-u', '--userid', type=int, help="Enter the userid")])
    def do_svd_based_recommendation(self,args, opts):
        if not (opts.noOfItems):
            raise Exception('mandatory parameters no of items missing')
        if not (opts.userid):
            raise Exception('mandatory parameters userid missing')
        if not(opts.refresh) or opts.refresh != "yes":
            opts.refresh="no"
        if int(opts.noOfItems):
            svd_on_transaction_data.generate_ratings(opts.userid, opts.noOfItems, opts.refresh)
        else:
            print("Supply a valid value of no of items")


    def do_rehelp(self, args):
        print("""CLI provides the functionality of triggering the commands
==== Recommend your recommendation engine

The command is  \t knn_on_movielens -u <userid> -n 3 -r <"yes"or"no">
Use this to get the rating on movielens data set using K-nearest neighbour algorithm

The command is  \t svd_on_movielens -u <userid> -n 3 -r <"yes"or"no">
Use this to get the rating on movielens data set using Singular value decomposition algorithm

The command is  \t als_based_recommendation -u <userid> -n 3 -r <"yes"or"no">
Use this to get the rating on movielens data set using Alternating Least Square algorithm


The command is  \t svd_based_recommendation -u <userid> -n 3 -r <"yes"or"no">
Use this to get the rating on transaction data set using Singular value decomposition algorithm

==== Auto Completion feature

CLI comes with auto completion feature""")


    def help_re_shell(self):
        print("""***Welcome to the CLI for Recommendation Engine utility commands! This is an online help utility.
the command rehelp() offers a short introduction***""")

    def do_quit(self, args):
        """Quits the program."""
        print("Quitting.")
        quit()

    def preloop(self):
        self.help_re_shell()

def init_commands():
    prompt = InteractiveCommands()
    prompt.prompt = 'Search Engine cli> '
    prompt.cmdloop()
