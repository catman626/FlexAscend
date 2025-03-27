import torch

def loadRefRecord(f):
    return torch.load(f).data.to("cpu")

def loadRecord(f):
    return torch.load(f)

def comp(a, b):
    diff = a - b
    m = torch.max(diff).item()
    n = torch.min(diff).item()
    return max(abs(m), abs(n))

def testAttentionOutput():
	for i in range(12):
		a = loadRefRecord(f"ref/attn.{i}")
		b = loadRecord(f"my/attn.{i}")
		diff = comp(a, b)

		ai = loadRefRecord(f"ref/attnIn.{i}")
		bi = loadRecord(f"my/attnIn.{i}")
		diffIn = comp(ai, bi)
		
		q = loadRefRecord(f"ref/q.{i}")
		k = loadRefRecord(f"ref/k.{i}")
		v = loadRefRecord(f"ref/v.{i}")

		q1 = loadRecord(f"my/q.{i}")
		k1 = loadRecord(f"my/k.{i}")
		v1 = loadRecord(f"my/v.{i}")

		assert q.shape == q1.shape, f"q shape: {q.shape}, q1 shape: {q1.shape}"
		assert k.shape == k1.shape, f"k shape: {k.shape}, k1 shape: {k1.shape}"
		assert v.shape == v1.shape, f"v shape: {v.shape}, v1 shape: {v1.shape}"

		diffq, diffk, diffv = comp(q, q1), comp(k, k1), comp(v, v1) 

		mha = loadRefRecord(f"ref/mha.{i}")
		mha1 = loadRecord(f"my/mha.{i}")
		diffmha = comp(mha, mha1)

		residual = loadRefRecord(f"ref/residual.{i}")
		residual1 = loadRecord(f"my/residual.{i}")
		diffresidual = comp(residual, residual1)

		outproj = loadRefRecord(f"ref/outProj.{i}")
		outproj1 = loadRecord(f"my/outProj.{i}")
		diffoutProj = comp(outproj, outproj1)

		report = ""
		report += f"diff of attentionOut.{diff:.4f}"
		report += f", diff of attentionIn.{diffIn:.4f}"
		# report += f", diff of q.{diffq:.4f}, k.{diffk:.4f}, v.{diffv:.4f}"
		# report += f", mha : {diffmha:.4f}"
		report += f", outProj : {diffoutProj:.4f}"
		report += f", residual : {diffresidual:.4f}"

		print(report)

def testInputEmbed():
	a=loadRefRecord("ref/inputEmbed")
	b=loadRecord("my/inputEmbed")
	diff = comp(a, b)
	print(f"diff of inputEmbed output: {diff}")
    
# def testAttentionIn():
    
 #   return loadFlexRecord()

testAttentionOutput()
testInputEmbed()


