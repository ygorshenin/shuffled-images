#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
using namespace std;

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& vs) {
  bool first = true;
  for (const auto& v : vs) {
    if (!first)
      os << " ";
    os << v;
    first = false;
  }
  return os;
}

struct Context {
  vector<vector<double>> m_hpreds;
  vector<vector<double>> m_vpreds;
  vector<int> m_permutation;
  int m_size = 0;
  int m_total = 0;
};

template <typename T>
void SortUnique(vector<T>& vs) {
  sort(vs.begin(), vs.end());
  vs.erase(unique(vs.begin(), vs.end()), vs.end());
}

template <typename T>
size_t SetIntersectionSize(const vector<T>& lhs, const vector<T>& rhs) {
  size_t i = 0, j = 0;
  size_t size = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i] < rhs[j]) {
      ++i;
    } else if (lhs[i] > rhs[j]) {
      ++j;
    } else {
      ++size;
      ++i;
      ++j;
    }
  }
  return size;
}

double GetScore(int size, const vector<int>& expected, const vector<int>& actual) {
  using Pair = pair<int, int>;

  vector<Pair> hexp;
  vector<Pair> hact;
  for (int r = 0; r < size; ++r) {
    for (int c = 0; c + 1 < size; ++c) {
      hexp.emplace_back(expected[r * size + c], expected[r * size + c + 1]);
      hact.emplace_back(actual[r * size + c], actual[r * size + c + 1]);
    }
  }

  vector<Pair> vexp;
  vector<Pair> vact;
  for (int r = 0; r + 1 < size; ++r) {
    for (int c = 0; c < size; ++c) {
      vexp.emplace_back(expected[r * size + c], expected[(r + 1) * size + c]);
      vact.emplace_back(actual[r * size + c], actual[(r + 1) * size + c]);
    }
  }

  const double a = SetIntersectionSize(hexp, hact) + SetIntersectionSize(vexp, vact);
  const double b = 2 * size * (size - 1);
  return a / b;
}

struct Piece {
  int m_row = 0;
  int m_col = 0;
  int m_piece = 0;
};

struct BoundingBox {
  int NumRows() const {
    assert(m_minRow <= m_maxRow);
    return m_maxRow - m_minRow;
  }

  int NumCols() const {
    assert(m_minCol <= m_maxCol);
    return m_maxCol - m_minCol;
  }

  int m_minRow = 0;
  int m_minCol = 0;
  int m_maxRow = 0;
  int m_maxCol = 0;
};

vector<int> Solve(const Context& ctx) {
  vector<int> result(ctx.m_total);
  iota(result.begin(), result.end(), 0);
  return result;
}

int main() {
  ios_base::sync_with_stdio(false);

  Context ctx;
  cin >> ctx.m_size;

  ctx.m_total = ctx.m_size * ctx.m_size;

  ctx.m_hpreds.assign(ctx.m_total, vector<double>(ctx.m_total));
  for (int i = 0; i < ctx.m_total; ++i) {
    for (int j = 0; j < ctx.m_total; ++j)
      cin >> ctx.m_hpreds[i][j];
  }
  ctx.m_vpreds.assign(ctx.m_total, vector<double>(ctx.m_total));
  for (int i = 0; i < ctx.m_total; ++i) {
    for (int j = 0; j < ctx.m_total; ++j)
      cin >> ctx.m_vpreds[i][j];
  }

  ctx.m_permutation.assign(ctx.m_total, 0);
  for (int i = 0; i < ctx.m_total; ++i)
    cin >> ctx.m_permutation[i];

  const auto result = Solve(ctx);
  if (result.size() != ctx.m_total) {
    cerr << "Wrong result size: expected " << ctx.m_total << ", actual: " << result.size() << endl;
    cerr << "Result permutation: " << result << endl;
    exit(EXIT_FAILURE);
  }
  vector<bool> used(ctx.m_total);
  for (const auto& r : result) {
    if (r < 0 || r >= ctx.m_total) {
      cerr << "Incorrect value " << r << " in result permutation: " << result;
      exit(EXIT_FAILURE);
    }
    if (used[r]) {
      cerr << "Duplicate value " << r << " in result permutation: " << result;
      exit(EXIT_FAILURE);
    }
  }

  cerr << "Result score: " << GetScore(ctx.m_size, /* expected= */ ctx.m_permutation, /* actual= */ result) << endl;
  cout << result << endl;
  exit(EXIT_SUCCESS);
}
