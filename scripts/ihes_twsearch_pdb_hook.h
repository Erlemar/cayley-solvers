// Optional, puzzle-checked additive corner / middle-slice lower bound.
// Compiled into an isolated copy of twsearch's solve.cpp by 31_build_ihes_hook.py.
#include <array>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
namespace ihes_pdb {
static std::vector<unsigned char> distances;
static std::vector<unsigned char> sum_distances;
static std::array<unsigned char,46656> mid;
static bool attempted=false;
static std::unordered_map<std::string,std::unordered_map<ull,float>> opening_scores;
static std::string root_key(setval s){
  static const char* digits="0123456789abcdef";std::string key;key.reserve(104);
  for(int i=0;i<52;i++){key+=digits[s.dat[i]>>4];key+=digits[s.dat[i]&15];}return key;
}
static int key(const unsigned char *p){int k=0;for(int i=0;i<6;i++)k=k*6+p[i];return k;}
static void init(const puzdef &pd){
  if(attempted)return;attempted=true;
  const char *root=std::getenv("IHES_CORNER_PDB_DIR");if(!root)return;
  int unit_moves=0;for(auto&m:pd.moves)unit_moves+=m.cost==1;
  if(pd.setdefs.size()!=3||pd.totsize!=52||unit_moves!=18)
    throw std::runtime_error("IHES PDB: wrong puzzle dimensions or move metric");
  const int ns[]={8,12,6},mods[]={3,2,4},offsets[]={0,16,40};
  for(int i=0;i<3;i++)if(pd.setdefs[i].size!=ns[i]||pd.setdefs[i].omod!=mods[i]||pd.setdefs[i].off!=offsets[i])
    throw std::runtime_error("IHES PDB: orbit convention mismatch");
  std::ifstream mf(std::string(root)+"/moves.txt");if(!mf)throw std::runtime_error("IHES PDB: missing moves.txt");
  for(int j=0;j<18;j++){
    std::string name;mf>>name;if(name[0]=='-')name=name.substr(1)+"'";
    std::array<int,52> a;for(int&v:a)mf>>v;
    const moove *found=nullptr;for(auto&m:pd.moves)if(m.name==name)found=&m;
    if(!found)throw std::runtime_error("IHES PDB: missing move "+name);
    for(int k=0;k<52;k++)if(a[k]!=found->pos.dat[k])throw std::runtime_error("IHES PDB: incompatible move "+name);
  }
  distances.resize(40320*2187);
  std::ifstream f(std::string(root)+"/corners_v1.bin",std::ios::binary);
  f.read(reinterpret_cast<char*>(distances.data()),distances.size());
  if(!f||f.peek()!=EOF||distances[0]!=0)throw std::runtime_error("IHES PDB: bad corner cache");
  if(!std::getenv("IHES_DISABLE_CENTER_SUM")){
    std::ifstream sf(std::string(root)+"/corners_center_sum_v1.bin",std::ios::binary);
    if(sf){sum_distances.resize(40320*2187*2);sf.read(reinterpret_cast<char*>(sum_distances.data()),sum_distances.size());
      if(!sf||sf.peek()!=EOF||sum_distances[0]!=0)throw std::runtime_error("IHES PDB: bad corner / center sum cache");}
  }
  mid.fill(255);std::vector<std::array<unsigned char,6>> q{{{0,1,2,3,4,5}}};mid[key(q[0].data())]=0;
  for(size_t i=0;i<q.size();i++)for(auto&m:pd.moves){
    if(m.cost!=1)continue;
    std::array<unsigned char,6>b;for(int k=0;k<6;k++)b[k]=q[i][m.pos.dat[40+k]];
    int j=key(b.data());if(mid[j]==255){mid[j]=mid[key(q[i].data())]+1;q.push_back(b);}
  }
  if(q.size()!=24)throw std::runtime_error("IHES PDB: center group mismatch");
  std::cout<<"IHES additive corner PDB loaded; 18 moves checked"<<std::endl;
  if(!sum_distances.empty())std::cout<<"IHES center orientation sum bound enabled"<<std::endl;
  if(const char *rankfile=std::getenv("IHES_PREFIX_RANK_FILE")){
    std::ifstream rf(rankfile);if(!rf)throw std::runtime_error("IHES missing opening ranks");
    std::string root;size_t n;while(rf>>root>>n){auto& scores=opening_scores[root];for(size_t i=0;i<n;i++){ull code;float score;rf>>code>>score;if(!rf)throw std::runtime_error("IHES malformed opening ranks");scores[code]=score;}}
    std::cout<<"IHES opening ranks loaded for "<<opening_scores.size()<<" roots"<<std::endl;
  }
}
static void order_chunks(setval root,std::vector<ull>& chunks){
  if(opening_scores.empty())return;auto r=opening_scores.find(root_key(root));
  if(r==opening_scores.end()){std::cout<<"IHES root has no opening ranks; original order retained"<<std::endl;return;}
  const auto& scores=r->second;std::vector<std::pair<float,ull>> order;size_t matched=0;
  for(ull c:chunks){auto it=scores.find(c);if(it!=scores.end())matched++;order.push_back({it==scores.end()?1e9f:it->second,c});}
  std::stable_sort(order.begin(),order.end(),[](auto&a,auto&b){return a.first<b.first;});
  for(size_t i=0;i<chunks.size();i++)chunks[i]=order[i].second;
  std::cout<<"IHES ranked "<<matched<<" of "<<chunks.size()<<" opening subtrees; all retained"<<std::endl;
}
static inline int lower(const setval s){
  unsigned seen=0;int p=0,o=0;
  for(int i=0;i<8;i++){unsigned v=s.dat[i];p=p*(8-i)+v-__builtin_popcount(seen&((1u<<v)-1));seen|=1u<<v;}
  for(int i=0;i<7;i++)o=3*o+s.dat[8+i];
  int c=p*2187+o,h=distances[c];
  if(!sum_distances.empty()){int sum=0;for(int i=46;i<52;i++)sum+=s.dat[i];h=sum_distances[c*2+(sum%4)/2];}
  return h+mid[key(s.dat+40)];
}
}
