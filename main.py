import streamlit as st
import hashlib
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def quasi_arbitrage():
    st.title('Model: "Sports Betting as an Alternative Investment"')


    np.random.seed(10)
    p = np.genfromtxt(r'p.csv')
    o = np.genfromtxt(r'o.csv')
    k = np.genfromtxt(r'k.csv')
    r = np.genfromtxt(r'r.csv')
    outcome = 1*(r>1)
    
    st.header('True Run')
    @st.cache_resource()
    def true_versus_sim():
        rand_probs = np.random.random((1000,len(p)))

        st.write('True run in black, simulatated runs in grey, where these simulated runs are created using the p, o and k arrays from the true run. Ideally, the true run would be close to the middle of the distribution of simulated runs, as this would suggest a calibrated model. Top panel shows with Kelly staking, natural logarithm to make visualisation possible; bottom panel shows with linear staking.')
        fig = plt.figure()
        ax = fig.add_subplot(211)
        result_cumprod = np.where(rand_probs <= p, 1 + k * (o - 1), 1 - k)
        log_cumprod = np.log(np.cumprod(result_cumprod, axis=1))
        ax.plot(np.arange(len(p)), log_cumprod.T, color='0.8')
        log_cumprod_r = np.log(np.cumprod(r))
        ax.plot(np.arange(len(p)), log_cumprod_r, color='black', label=f'Percentile: {round(1 / 100 * percentileofscore(log_cumprod[:, -1], log_cumprod_r[-1]), 3)}')
        ax.hlines(0,xmin=ax.set_xlim()[0],xmax=len(p),color='black',linewidth=0.75)
        ax.set(xlabel=r'n',ylabel=r'ln(C_n)',title='Geometric')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left')
        ax = fig.add_subplot(212)
        result_cumsum = np.where(rand_probs <= p, o - 1, -1)
        cumsum_result = np.cumsum(result_cumsum, axis=1)
        ax.plot(np.arange(len(p)), cumsum_result.T, color='0.8')
        cumsum_r = np.cumsum(np.where(r > 1, o - 1, -1))
        ax.plot(np.arange(len(p)), cumsum_r, color='black', label=f'Percentile: {round(1 / 100 * percentileofscore(cumsum_result[:, -1], cumsum_r[-1]), 3)}')
        ax.hlines(0,xmin=ax.set_xlim()[0],xmax=len(p),color='black',linewidth=0.75)
        ax.set(xlabel=r'n',ylabel=r'pi_n',title='Linear')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
    true_versus_sim()

    st.header('Samples of True Run')
    st.write('Select n. Figure shows 1000 samples of selected bets from the true run. No simulation here, simply taking small samples of the true run. Important to consider how a large-sample-size model performs over smaller samples.')
    @st.cache_resource()
    def sim_sample(n):
        l = np.where(r>1,o-1,-1)
        its = 1000
        sub_geometric = np.array([[1] + list(np.random.choice(r,n,replace=False)) for _ in range(its)])
        cumprod_sub = np.cumprod(sub_geometric,axis=1)
        sub_linear = np.array([[0] + list(np.random.choice(l,n,replace=False)) for _ in range(its)])
        cumsum_sub = np.cumsum(sub_linear,axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(np.arange(n+1), np.log(cumprod_sub.T), color='0.8')
        ax.text(ax.set_xlim()[1]/8, ax.set_ylim()[1]*0.75,ha='center',s=f'Mean: {round(np.mean(np.log(cumprod_sub).T[-1]),3)}\nMedian: {round(np.median(np.log(cumprod_sub).T[-1]),3)}\nStandard Deviation: {round(np.std(np.log(cumprod_sub).T[-1]),3)}\nP(Losses): {round(percentileofscore(np.log(cumprod_sub).T[-1],0),3)}')
        ax.hlines(0,xmin=ax.set_xlim()[0],xmax=n,color='black',linewidth=0.75)
        ax.set(xlabel=r'n',ylabel=r'ln(C_n)',title='Geometric')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax = fig.add_subplot(212)
        ax.plot(np.arange(n+1), cumsum_sub.T, color='0.8')
        ax.text(ax.set_xlim()[1]/8, ax.set_ylim()[1]*0.75,ha='center',s=f'Mean: {round(np.mean(cumsum_sub.T[-1]),3)}\nMedian: {round(np.median(cumsum_sub.T[-1]),3)}\nStandard Deviation: {round(np.std(cumsum_sub.T[-1]),3)}\nP(Losses): {round(percentileofscore(cumsum_sub.T[-1],0),3)}')
        ax.hlines(0,xmin=ax.set_xlim()[0],xmax=n,color='black',linewidth=0.75)
        ax.set(xlabel=r'n',ylabel=r'pi_n',title='Linear')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)    
    n = st.selectbox('n',options=[i for i in [10,50,100,500,1000] if i < len(p)/2],index=2)
    sim_sample(n)

    
    from hypothesis_tests import poi_bin_test, geom_r_test, lin_r_test, brier_test, pcs_test
    
    @st.cache_data()
    def hypothesis_testing():
        st.header('Hypothesis Testing')
        
        st.write(f'#### Geometric-Returns Test')
        st.write('By central limit theorem, assume log cumulative geometric returns are normally distributed. Two-sided p-value.')
        gr = geom_r_test(r)
        st.write(f'Test-Statistic: {round(gr[0],3)}, P-Value: {gr[1]:.3g}')
        
        st.write(f'#### Linear-Returns Test')
        st.write('By central limit theorem, assume linear returns are normally distributed. Two-sided p-value.')
        lr = lin_r_test(o,outcome)
        st.write(f'Test-Statistic: {round(lr[0],3)}, P-Value: {lr[1]:.3g}')
        
        st.write(f'#### Poisson-Binomial Test')
        st.write('Given our vector of probabilities p, what is the probability of seeing the observed number of wins. The null hypothesis is a calibrated forecasting model.')
        pbt = poi_bin_test(p,int(np.sum(outcome)))
        st.write(f'P-Value: {pbt[1]:.3g}')
        
        st.write(f'#### Brier-Score Test')
        bs = brier_test(p,outcome,5)
        if bs[-1] == 1:
            st.write('Have shown that Brier scores are normally distributed under null hypothesis of perfect calibration. Ideally, the test-statistic is close to zero.')
            st.write('Test is applicable with our given p.')
            st.write(f'Test-Statistic: {round(bs[0],3)}, P-Value: {bs[1]:.3g}')
        else:
            st.write('Test is not applicable with our given p.')
        
        st.write(f"#### Pearson's Chi-Squared Test")
        st.write('Bin probabilities such that they contain the same number of elements. Compare expected frequency to observed frequency in each bin. Test-statistic is chi-squared distributed. The null hypothesis is a calibrated forecasting model.')
        pbt = pcs_test(p, outcome)
        st.write(f'Test-Statistic: {round(pbt[0],3)}, P-Value: {pbt[1]:.3g}')
    hypothesis_testing()

    st.header('Placed Bets, Details')
    @st.cache_data()
    def bets_table():
        def price_difference(p_mean, p_min):
            return np.log(1 / (1 - p_mean)) - np.log(1 / (1 - p_min))
        price_difference = np.vectorize(price_difference)
        l_pd = price_difference(p, 1/o)

        table = pd.DataFrame(np.array([[f'{i:.3g}' for i in p],o,[f'{i:.3g}' for i in k],[f'{i:.3g}' for i in l_pd],outcome,['' for i in range(len(p))],['' for i in range(len(p))]]).T,columns=['Probability', 'Odds', 'Kelly Fraction', 'Price Difference', 'Outcome','Bookmaker','Date'])
        st.dataframe(table)
    bets_table()


    
#Password Verification
def verify_password(input_password, stored_hashed_password):
    return hashlib.sha256(input_password.encode()).hexdigest() == stored_hashed_password

def check_password():
    if verify_password(st.session_state['password'], st.secrets["PASSWORD_HASH"]):
        st.session_state['authenticated'] = True
    else:
        st.session_state['authenticated'] = False
        st.error('Incorrect password')

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    st.text_input("Password", type='password', on_change=check_password, key='password')
else:
    if 'page' not in st.session_state:
        st.session_state.page = 'main_page'
    
    if st.session_state.page == 'main_page':
        selection = 'Existing Sentence Database'

        st.sidebar.title('Navigation')
        options = ['Model: "Sports Betting as an Alternative Investment"', 'Model: Momentum Win Market', 'Model: Orangutan', 'Model: Money for Jam']
        selection = st.sidebar.radio("Go to", options,)
        if selection == 'Model: "Sports Betting as an Alternative Investment"':
            quasi_arbitrage()
        else:
            def coming_soon():
                st.title('Coming Soon')
            coming_soon()
    
    