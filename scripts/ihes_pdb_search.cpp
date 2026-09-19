// IHES exact coordinate search. All projections use the official 18 quarter turns.
// Full corners count only outer moves, allowing an additive middle-slice bound.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace std;
using Clock = chrono::steady_clock;
constexpr int NM=18, NC=2187, NE=2048, NT=98304, NP=40320;
struct Move {string name; array<int,8> cp,co; array<int,12> ep,eo; array<int,6> tp,to;};
array<Move,NM> moves;
array<array<uint16_t,NM>,NP> pm;
array<array<uint16_t,NM>,NC> cm;
array<array<uint16_t,NM>,NE> em;
vector<array<uint32_t,NM>> center_moves(NT);
vector<array<int,6>> center_perms;
unordered_map<int,int> center_ids;
array<int,24> middle_dist;
vector<uint8_t> pc, pct, pet, pcs;
array<int,NM> center_sum_delta;
array<int,NP> corner_parity;
string dir;
int center_key(const array<int,6>& a) {int r=0; for(int v:a)r=r*6+v; return r;}
template<size_t N> int rank_perm(const array<int,N>& p) {
    int r=0; for(int i=0;i<int(N);i++){int k=0; for(int j=i+1;j<int(N);j++)k+=p[j]<p[i]; r=r*(N-i)+k;} return r;
}
template<size_t N> array<int,N> unrank_perm(int r) {
    array<int,N> digits{},p{},avail{}; iota(avail.begin(),avail.end(),0);
    for(int i=N-1;i>=0;i--){digits[i]=r%(N-i);r/=N-i;}
    int n=N;for(int i=0;i<int(N);i++){p[i]=avail[digits[i]];for(int j=digits[i];j<n-1;j++)avail[j]=avail[j+1];n--;}return p;
}
template<size_t N> int rank_ori(const array<int,N>& a,int base,bool constrained) {
    int r=0;for(int i=0;i<int(N)-int(constrained);i++)r=r*base+a[i];return r;
}
template<size_t N> array<int,N> unrank_ori(int r,int base,bool constrained) {
    array<int,N>a{};int sum=0;for(int i=int(N)-1-int(constrained);i>=0;i--){a[i]=r%base;r/=base;sum+=a[i];}
    if(constrained)a[N-1]=(base-sum%base)%base;return a;
}
void setup(){
    ifstream f(dir+"/moves.txt");if(!f)throw runtime_error("missing moves");
    for(auto&m:moves){f>>m.name;for(auto*a:{&m.cp,&m.co})for(auto&v:*a)f>>v;for(auto*a:{&m.ep,&m.eo})for(auto&v:*a)f>>v;for(auto*a:{&m.tp,&m.to})for(auto&v:*a)f>>v;}
    array<int,6> ident{0,1,2,3,4,5};center_perms.push_back(ident);center_ids[center_key(ident)]=0;middle_dist.fill(99);middle_dist[0]=0;
    for(size_t q=0;q<center_perms.size();q++)for(int m=0;m<NM;m++){
        array<int,6> next{};for(int i=0;i<6;i++)next[i]=center_perms[q][moves[m].tp[i]];
        int key=center_key(next);if(!center_ids.count(key)){int n=center_perms.size();center_ids[key]=n;center_perms.push_back(next);middle_dist[n]=middle_dist[q]+1;}
    }
    if(center_perms.size()!=24)throw runtime_error("center permutation group is not 24");
    for(int p=0;p<NP;p++){auto a=unrank_perm<8>(p);if(rank_perm(a)!=p)throw runtime_error("rank");int inv=0;for(int i=0;i<8;i++)for(int j=i+1;j<8;j++)inv+=a[i]>a[j];corner_parity[p]=inv%2;for(int m=0;m<NM;m++){array<int,8>b{};for(int i=0;i<8;i++)b[i]=a[moves[m].cp[i]];pm[p][m]=rank_perm(b);}}
    for(int m=0;m<NM;m++){center_sum_delta[m]=accumulate(moves[m].to.begin(),moves[m].to.end(),0)%4;if((m/2)%3==1&&center_sum_delta[m])throw runtime_error("middle changes total center orientation");}
    for(int o=0;o<NC;o++){auto a=unrank_ori<8>(o,3,true);for(int m=0;m<NM;m++){array<int,8>b{};for(int i=0;i<8;i++)b[i]=(a[moves[m].cp[i]]+moves[m].co[i])%3;cm[o][m]=rank_ori(b,3,true);}}
    for(int o=0;o<NE;o++){auto a=unrank_ori<12>(o,2,true);for(int m=0;m<NM;m++){array<int,12>b{};for(int i=0;i<12;i++)b[i]=a[moves[m].ep[i]]^moves[m].eo[i];em[o][m]=rank_ori(b,2,true);}}
    for(int t=0;t<NT;t++){auto a=unrank_ori<6>(t%4096,4,false);auto perm=center_perms[t/4096];for(int m=0;m<NM;m++){array<int,6>b{},bp{};for(int i=0;i<6;i++){b[i]=(a[moves[m].tp[i]]+moves[m].to[i])%4;bp[i]=perm[moves[m].tp[i]];}center_moves[t][m]=center_ids.at(center_key(bp))*4096+rank_ori(b,4,false);}}
    // Every coordinate transition must invert exactly.
    for(int m=0;m<NM;m++){
        for(int p=0;p<NP;p++)if(pm[pm[p][m]][m^1]!=p)throw runtime_error("cp inverse");
        for(int o=0;o<NC;o++)if(cm[cm[o][m]][m^1]!=o)throw runtime_error("co inverse");
        for(int o=0;o<NE;o++)if(em[em[o][m]][m^1]!=o)throw runtime_error("eo inverse");
        for(int t=0;t<NT;t++)if(center_moves[center_moves[t][m]][m^1]!=uint32_t(t))throw runtime_error("center inverse");
    }
    cout<<"coordinate transitions verified"<<endl;
}
template<class Next> void build(vector<uint8_t>&d,const string&name,uint32_t size,Next next,const vector<int>&ms){
    string path=dir+"/"+name+".bin";
    if(filesystem::exists(path)&&filesystem::file_size(path)==size){d.resize(size);ifstream f(path,ios::binary);f.read((char*)d.data(),size);if(!f)throw runtime_error("cache read");cout<<"loaded "<<name<<" "<<size<<endl;return;}
    d.assign(size,255);vector<uint32_t>q;q.reserve(size);q.push_back(0);d[0]=0;size_t head=0,end=1;auto start=Clock::now();
    for(int depth=0;head<q.size();depth++){
        while(head<end){uint32_t x=q[head++];for(int m:ms){uint32_t y=next(x,m);if(d[y]==255){d[y]=depth+1;q.push_back(y);}}}
        cout<<"build "<<name<<" depth="<<depth<<" reached="<<q.size()<<" seconds="<<chrono::duration<double>(Clock::now()-start).count()<<endl;end=q.size();
    }
    if(q.size()!=size)throw runtime_error("projection not connected: "+name+" "+to_string(q.size()));
    ofstream f(path+".tmp",ios::binary);f.write((char*)d.data(),d.size());f.close();if(!f)throw runtime_error("cache write");filesystem::rename(path+".tmp",path);
}
void databases(bool with_eo){
    vector<int>all(NM),outer;iota(all.begin(),all.end(),0);for(int m:all)if((m/2)%3!=1)outer.push_back(m);
    build(pc,"corners_v1",uint32_t(NP)*NC,[](uint32_t x,int m){return uint32_t(pm[x/NC][m])*NC+cm[x%NC][m];},outer);
    build(pct,"corner_orientation_centers_v1",uint32_t(NC)*NT,[](uint32_t x,int m){return uint32_t(cm[x/NT][m])*NT+center_moves[x%NT][m];},all);
    if(with_eo)build(pet,"edge_orientation_centers_v1",uint32_t(NE)*NT,[](uint32_t x,int m){return uint32_t(em[x/NT][m])*NT+center_moves[x%NT][m];},all);
}
void sum_database(){
    vector<int>outer;for(int m=0;m<NM;m++)if((m/2)%3!=1)outer.push_back(m);
    build(pcs,"corners_center_sum_v1",uint32_t(NP)*NC*2,[](uint32_t x,int m){uint32_t c=x/2;int p=c/NC,o=c%NC;int sum=(x%2)*2+corner_parity[p];int ns=(sum+center_sum_delta[m])%4;return (uint32_t(pm[p][m])*NC+cm[o][m])*2+ns/2;},outer);
}
struct State {int cp=0,co=0,eo=0,ct=0; array<uint8_t,12> ep{};};
struct Query{int pid,len;State s;};
vector<Query> read_queries(){
    ifstream f(dir+"/queries.txt");vector<Query>out;int pid,len;
    while(f>>pid>>len){array<int,8>cp,co;array<int,12>ep,eo;array<int,6>tp,to;for(auto*a:{&cp,&co})for(auto&v:*a)f>>v;for(auto*a:{&ep,&eo})for(auto&v:*a)f>>v;for(auto*a:{&tp,&to})for(auto&v:*a)f>>v;
        State s;s.cp=rank_perm(cp);s.co=rank_ori(co,3,true);s.eo=rank_ori(eo,2,true);s.ct=center_ids.at(center_key(tp))*4096+rank_ori(to,4,false);copy(ep.begin(),ep.end(),s.ep.begin());out.push_back({pid,len,s});}
    return out;
}
inline int heuristic(const State&s,int mode=3){
    int h=pc[s.cp*NC+s.co]+middle_dist[s.ct/4096];
    if(mode>=2)h=max(h,int(pct[uint32_t(s.co)*NT+s.ct]));
    if(mode>=3&&!pet.empty())h=max(h,int(pet[uint32_t(s.eo)*NT+s.ct]));
    return h;
}
inline State step(const State&s,int m){State n;n.cp=pm[s.cp][m];n.co=cm[s.co][m];n.eo=em[s.eo][m];n.ct=center_moves[s.ct][m];for(int i=0;i<12;i++)n.ep[i]=s.ep[moves[m].ep[i]];return n;}
inline bool solved(const State&s){if(s.cp||s.co||s.eo||s.ct)return false;for(int i=0;i<12;i++)if(s.ep[i]!=i)return false;return true;}
bool allowed(int m,int last,int previous){
    if(last<0)return true;
    if(m==(last^1))return false;
    // Parallel layers commute: sort by layer, and retain one sign for half turns.
    if(m/6==last/6){if(m/2<last/2)return false;if(m/2==last/2){if(m&1)return false;if(previous>=0&&previous/2==m/2)return false;}}
    return true;
}
struct Search {
    atomic<bool>stop{false},hit{false};atomic<uint64_t>nodes{0};Clock::time_point deadline;
    vector<int>answer;int mode=3;int threads=8;
    bool dfs(const State&s,int rem,int last,int prev,vector<int>&path,uint64_t&local){
        ++local;if((local&4095)==0){if(stop.load(memory_order_relaxed))return false;if(Clock::now()>deadline){stop=true;return false;}}
        if(heuristic(s,mode)>rem)return false;
        if(rem==0){if(!solved(s))return false;bool expected=false;if(hit.compare_exchange_strong(expected,true))answer=path;stop=true;return true;}
        for(int m=0;m<NM;m++)if(allowed(m,last,prev)){path.push_back(m);if(dfs(step(s,m),rem-1,m,last,path,local))return true;path.pop_back();}
        return false;
    }
    void run(const Query&q,int maxdepth,double seconds){
        deadline=Clock::now()+chrono::milliseconds(int64_t(seconds*1000));auto start=Clock::now();
        int low=heuristic(q.s,mode);if((low&1)!=(q.len&1))low++;
        for(int depth=low;depth<=maxdepth&&!stop;depth+=2){
            struct Task{State s;vector<int>p;};vector<Task>tasks{{q.s,{}}};
            int split=min(2,depth);for(int k=0;k<split;k++){vector<Task>next;for(auto&t:tasks)for(int m=0;m<NM;m++)if(allowed(m,t.p.empty()?-1:t.p.back(),t.p.size()<2?-1:t.p[t.p.size()-2])){auto path=t.p;path.push_back(m);next.push_back({step(t.s,m),path});}tasks=move(next);}
            atomic<size_t>idx{0};vector<thread>workers;for(int t=0;t<threads;t++)workers.emplace_back([&]{uint64_t local=0;while(!stop){size_t i=idx++;if(i>=tasks.size())break;auto&job=tasks[i];auto p=job.p;dfs(job.s,depth-split,p.empty()?-1:p.back(),p.size()<2?-1:p[p.size()-2],p,local);}nodes+=local;});for(auto&t:workers)t.join();
            cout<<"depth pid="<<q.pid<<" d="<<depth<<" nodes="<<nodes<<" seconds="<<chrono::duration<double>(Clock::now()-start).count()<<endl;
        }
        cout<<"RESULT {\"pid\":"<<q.pid<<",\"incumbent\":"<<q.len<<",\"max_depth\":"<<maxdepth<<",\"heuristic\":"<<heuristic(q.s,mode)<<",\"verdict\":\""<<(hit?"hit":stop?"timeout":"none")<<"\",\"nodes\":"<<nodes<<",\"seconds\":"<<chrono::duration<double>(Clock::now()-start).count()<<",\"path\":\"";
        if(hit){State s=q.s;for(size_t i=0;i<answer.size();i++){if(i)cout<<'.';cout<<moves[answer[i]].name;s=step(s,answer[i]);}if(!solved(s))throw runtime_error("internal replay failed");}cout<<"\"}"<<endl;
    }
};
int main(int argc,char**argv){try{
    dir="data/ihes_pdb";string command="build";int pid=-1,limit=-1,threads=8,mode=3;double seconds=60;
    for(int i=1;i<argc;i++){string a=argv[i];auto val=[&]{if(++i>=argc)throw runtime_error("missing argument");return string(argv[i]);};if(a=="--dir")dir=val();else if(a=="--command")command=val();else if(a=="--pid")pid=stoi(val());else if(a=="--depth")limit=stoi(val());else if(a=="--threads")threads=stoi(val());else if(a=="--mode")mode=stoi(val());else if(a=="--seconds")seconds=stod(val());else throw runtime_error("unknown argument: "+a);}
    setup();if(command=="build-sum"){sum_database();return 0;}
    if(command=="build-additive"){
      vector<int>outer;for(int m=0;m<NM;m++)if((m/2)%3!=1)outer.push_back(m);
      build(pc,"corners_v1",uint32_t(NP)*NC,[](uint32_t x,int m){return uint32_t(pm[x/NC][m])*NC+cm[x%NC][m];},outer);
      sum_database();return 0;
    }
    databases(mode>=3);auto qs=read_queries();
    if(command=="build")return 0;
    if(command=="audit"){for(auto&q:qs)cout<<"AUDIT "<<q.pid<<' '<<q.len<<' '<<heuristic(q.s,1)<<' '<<heuristic(q.s,2)<<' '<<heuristic(q.s,3)<<endl;return 0;}
    if(command=="selftest"){
        mt19937 rng(31415);State root;iota(root.ep.begin(),root.ep.end(),0);
        for(int rep=0;rep<1000;rep++){State s=root;for(int k=1;k<=30;k++){int m=rng()%NM;s=step(s,m);if(heuristic(s)>k)throw runtime_error("inadmissible walk bound");for(int n=0;n<NM;n++){State t=step(s,n);if(abs(heuristic(s)-heuristic(t))>1)throw runtime_error("inconsistent heuristic");}}}
        for(int rep=0;rep<12;rep++){State s=root;int depth=4+rep%5;for(int j=0;j<depth;j++)s=step(s,rng()%NM);Search search;search.threads=threads;search.run({-rep-1,depth,s},depth,30);if(!search.hit)throw runtime_error("positive solve control failed");}
        cout<<"SELFTEST PASSED"<<endl;return 0;
    }
    for(auto&q:qs)if(q.pid==pid){Search search;search.threads=threads;search.mode=mode;search.run(q,limit<0?q.len-2:limit,seconds);return 0;}
    throw runtime_error("pid not found");
}catch(const exception&e){cerr<<"ERROR "<<e.what()<<endl;return 1;}}
