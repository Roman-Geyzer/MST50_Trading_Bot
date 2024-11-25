#run_main.py

from MST50.main import main


run_mode = ['transfer']
# back_test, demo, live, on_hold, dev, optimize, transfer


# Determine if we are in backtesting mode
BACKTEST_MODE = True

#To see profile results, run the following command in the terminal:
# snakeviz backtest_profile.prof

################# Profiling #################


import cProfile
import pstats
import io

def profile_backtest():
    pr = cProfile.Profile()
    pr.enable()
    try:
        main(run_mode,BACKTEST_MODE)
    except KeyboardInterrupt:
        print("Backtest interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'  # You can also use 'time' or other sort options
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print("documentated the profile in backtest_profile.prof")
        pr.dump_stats('backtest_profile.prof')

################# end Profiling #################

if __name__ == "__main__":
    if BACKTEST_MODE :
        main(run_mode,BACKTEST_MODE)
    else:
        profile_backtest()